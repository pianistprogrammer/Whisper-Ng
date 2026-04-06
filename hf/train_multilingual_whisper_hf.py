"""
Fine-tuning OpenAI Whisper (small) on 4 Nigerian languages using Hugging Face Transformers.

Languages:
  - Yoruba  (yor_tts)
  - Hausa   (hau_tts)
  - Igbo    (ibo_tts)
  - Pidgin English (pcm_tts)

Requirements:
    pip install transformers datasets accelerate evaluate jiwer tensorboard

Run:
    python train_yoruba_whisper_hf.py

Outputs (all saved to OUTPUT_DIR):
    - Model checkpoints
    - Training logs
    - Evaluation metrics
"""

import torch
import sys
import os

# Apple Silicon: MPS unified memory is managed by the OS.
# The default PyTorch watermark blocks valid allocations long before the system is full.
# Setting it to 0.0 removes the limit and lets macOS handle memory pressure naturally.
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

def _get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

DEVICE = _get_device()

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Union
import evaluate
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe for training scripts
import matplotlib.pyplot as plt
import numpy as np

from datasets import load_dataset, concatenate_datasets, DatasetDict, Audio
from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)

# ---------------------------------------------------------------------------
# Tee: mirror all stdout  to a timestamped log file
# ---------------------------------------------------------------------------

class _Tee:
    """Write every print() call to both the terminal and a log file."""
    def __init__(self, log_path: Path):
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(log_path, "w", buffering=1, encoding="utf-8")  # line-buffered
        self._stdout = sys.stdout
        sys.stdout = self

    def write(self, data):
        self._stdout.write(data)
        self._file.write(data)

    def flush(self):
        self._stdout.flush()
        self._file.flush()

    def close(self):
        sys.stdout = self._stdout
        self._file.close()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_NAME = "openai/whisper-small"
DATASET_NAME = "google/WaxalNLP"
TASK = "transcribe"

# All 4 Nigerian language configs to combine
LANGUAGE_CONFIGS = [
    {"config": "yor_tts", "language": "yoruba"},
    {"config": "hau_tts", "language": "hausa"},
    {"config": "ibo_tts", "language": "igbo"},
    {"config": "pcm_tts", "language": "english"},  # Pidgin uses English tokenizer
]

RANDOM_SEED = 42
OUTPUT_DIR = "./whisper-small-nigerian"
SAMPLE_RATE = 16000

# Trackio project — all runs from this script land in this project
TRACKIO_PROJECT = "whisper-ng-nigerian"

# ---------------------------------------------------------------------------
# Start logging — everything printed from here on goes to terminal + log file
# ---------------------------------------------------------------------------
# Training hyperparameters
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 16
LEARNING_RATE = 5e-6
WARMUP_STEPS = 500
MAX_STEPS = 5000
EVAL_STEPS = 1000
SAVE_STEPS = 1000
LOGGING_STEPS = 25
SMOOTH_WINDOW = 10   # window for rolling-average overlay on the step-loss plot

# Set to True to auto-resume from the latest checkpoint in OUTPUT_DIR,
# or set to a specific path e.g. "./whisper-small-nigerian/checkpoint-250"
RESUME_FROM_CHECKPOINT = False


_RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
_LOG_PATH = Path(OUTPUT_DIR) / f"training_log_{_RUN_TIMESTAMP}.txt"
_tee = _Tee(_LOG_PATH)
print(f"Training log: {_LOG_PATH.resolve()}")
print(f"Run started : {_RUN_TIMESTAMP}")
print(f"Model       : {MODEL_NAME}")
print(f"Max steps   : {MAX_STEPS}  |  Eval every {EVAL_STEPS}  |  Save every {SAVE_STEPS}")
print(f"Batch size  : {BATCH_SIZE}  |  Grad accum: {GRADIENT_ACCUMULATION_STEPS}  "
      f"→  effective batch = {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
print(f"LR          : {LEARNING_RATE}  |  Warmup: {WARMUP_STEPS} steps")
print(f"Device      : {DEVICE.upper()}")
if RESUME_FROM_CHECKPOINT:
    print(f"Resuming    : {RESUME_FROM_CHECKPOINT}")
print()


# ---------------------------------------------------------------------------
# Load Feature Extractor, Tokenizer and Processor
# ---------------------------------------------------------------------------

print("Loading feature extractor, tokenizer, and processor...")
feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL_NAME)
tokenizer = WhisperTokenizer.from_pretrained(MODEL_NAME, task=TASK)
processor = WhisperProcessor.from_pretrained(MODEL_NAME, task=TASK)

# ---------------------------------------------------------------------------
# Load and Prepare Dataset — all 4 languages
# ---------------------------------------------------------------------------

print("Loading datasets for all 4 languages...")
all_train, all_val, all_test = [], [], []

for lang_cfg in LANGUAGE_CONFIGS:
    cfg, lang = lang_cfg["config"], lang_cfg["language"]
    print(f"  Loading {lang.upper()} ({cfg})...", flush=True)
    ds = load_dataset(DATASET_NAME, cfg)
    ds = ds.cast_column("audio", Audio(sampling_rate=SAMPLE_RATE))
    print(f"    train={len(ds['train']):,}  val={len(ds['validation']):,}  test={len(ds['test']):,}")
    all_train.append(ds["train"])
    all_val.append(ds["validation"])
    all_test.append(ds["test"])

print("\nCombining and shuffling all languages...", flush=True)
combined_train = concatenate_datasets(all_train).shuffle(seed=RANDOM_SEED)
combined_val   = concatenate_datasets(all_val).shuffle(seed=RANDOM_SEED)
combined_test  = concatenate_datasets(all_test).shuffle(seed=RANDOM_SEED)

print(f"  Combined train : {len(combined_train):,}")
print(f"  Combined val   : {len(combined_val):,}")
print(f"  Combined test  : {len(combined_test):,}")

# Use train + val for training (maximise data), test for evaluation
print("\nBuilding final DatasetDict...")
common_voice = DatasetDict()
common_voice["train"] = concatenate_datasets([combined_train, combined_val]).shuffle(seed=RANDOM_SEED)
common_voice["test"]  = combined_test

# ---------------------------------------------------------------------------
# Prepare Data
# ---------------------------------------------------------------------------

def prepare_dataset(batch):
    """Prepare a single batch for training."""
    # Load and resample audio
    audio = batch["audio"]

    # Compute log-Mel input features from input audio array
    batch["input_features"] = feature_extractor(
        audio["array"],
        sampling_rate=audio["sampling_rate"]
    ).input_features[0]

    # FIX: Add special tokens (Whisper decoder prompts like <|startoftranscript|>)
    labels = tokenizer(batch["text"], add_special_tokens=True).input_ids

    # Truncate labels to max length (448 tokens for Whisper base decoder)
    # This prevents indexing errors during training
    max_label_length = 448
    if len(labels) > max_label_length:
        labels = labels[:max_label_length]

    batch["labels"] = labels

    return batch

print("Preprocessing dataset...")
common_voice = common_voice.map(
    prepare_dataset,
    remove_columns=common_voice.column_names["train"],
    num_proc=1
)

# ---------------------------------------------------------------------------
# Data Collator
# ---------------------------------------------------------------------------

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # Split inputs and labels since they have different lengths and need
        # different padding methods

        # First treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # Get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # Pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # Replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # If bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        
        # FIX: Explicitly set decoder_attention_mask
        # This fixes the warning: "attention mask is not set and cannot be inferred"
        # Especially important because pad_token_id == eos_token_id
        decoder_attention_mask = labels_batch.attention_mask
        if (labels_batch["input_ids"][:, 0] == self.decoder_start_token_id).all().cpu().item():
            decoder_attention_mask = decoder_attention_mask[:, 1:]
        batch["decoder_attention_mask"] = decoder_attention_mask

        return batch

# ---------------------------------------------------------------------------
# Load Pre-trained Model
# ---------------------------------------------------------------------------

print("Loading pre-trained Whisper model...")
model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
model.config.use_cache = False
model = model.to(DEVICE)

# gradient_checkpointing requires CUDA; skip on MPS/CPU to avoid errors
if DEVICE == "cuda":
    model.gradient_checkpointing_enable()

# Multilingual mode — do NOT force a single language token
model.generation_config.task = TASK
model.generation_config.forced_decoder_ids = None
model.generation_config.language = None
print("Full model (encoder + decoder) will be fine-tuned from step 1.")

# ---------------------------------------------------------------------------
# Evaluation Metrics
# ---------------------------------------------------------------------------

try:
    metric = evaluate.load("wer")
except AttributeError:
    # Fallback for huggingface_hub version incompatibility
    from jiwer import wer as compute_wer

    class WERMetric:
        def compute(self, predictions, references):
            return compute_wer(references, predictions)

    metric = WERMetric()

def compute_metrics(pred):
    """Compute WER metric."""
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # Replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # We do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}

# ---------------------------------------------------------------------------
# Training Arguments
# ---------------------------------------------------------------------------

training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    learning_rate=LEARNING_RATE,
    warmup_steps=WARMUP_STEPS,
    max_steps=MAX_STEPS,
    gradient_checkpointing=DEVICE == "cuda",  # not supported on MPS
    fp16=DEVICE == "cuda",
    bf16=False,                               # not supported by HF Trainer on MPS
    eval_strategy="steps",
    per_device_eval_batch_size=1,
    predict_with_generate=True,
    generation_max_length=128,
    save_steps=SAVE_STEPS,
    eval_steps=EVAL_STEPS,
    logging_steps=LOGGING_STEPS,
    report_to=["tensorboard", "trackio"],
    run_name=f"whisper-small-nigerian-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    project=TRACKIO_PROJECT,
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    save_total_limit=2,          # keep only the 2 best checkpoints on disk
    push_to_hub=False,  # Set to True if you want to push to Hub
    dataloader_pin_memory=False,
)

# ---------------------------------------------------------------------------
# Initialize Trainer
# ---------------------------------------------------------------------------

# Unfreeze encoder halfway through training so the full model can adapt
# once the decoder is already learning the target languages.
from transformers import TrainerCallback


class StepLoggerCallback(TrainerCallback):
    """Write one structured line per logging/eval event to stdout (captured by _Tee)."""
    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        step = state.global_step
        parts = [f"step={step:>5}"]
        for key in ["loss", "learning_rate", "grad_norm",
                    "eval_loss", "eval_wer",
                    "train_runtime", "train_samples_per_second", "train_steps_per_second"]:
            if key in logs:
                val = logs[key]
                if isinstance(val, float):
                    parts.append(f"{key}={val:.6g}")
                else:
                    parts.append(f"{key}={val}")
        print("  ".join(parts), flush=True)

print("Initializing trainer...")
data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id,
)

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=common_voice["train"],
    eval_dataset=common_voice["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[
        StepLoggerCallback(),
    ],
)

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Training Graphs
# ---------------------------------------------------------------------------

def save_training_graphs_hf(trainer, output_dir: Path):
    """Parse trainer.state.log_history and save 4 charts to a timestamped subdir."""
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    graphs_dir = output_dir / f"graphs_{timestamp}"
    graphs_dir.mkdir(parents=True, exist_ok=True)

    log_history = trainer.state.log_history
    train_logs  = [e for e in log_history if "loss" in e and "eval_loss" not in e]
    eval_logs   = [e for e in log_history if "eval_loss" in e]

    if not train_logs:
        print("  No training logs found — skipping graphs.")
        return

    step_numbers = [e["step"]  for e in train_logs]
    step_losses  = [e["loss"]  for e in train_logs]
    eval_steps   = [e["step"]      for e in eval_logs]
    eval_losses  = [e["eval_loss"] for e in eval_logs]
    eval_wers    = [e.get("eval_wer") for e in eval_logs]

    # Pre-compute smoothed curve (reused across charts)
    smoothed = smooth_x = None
    if len(step_losses) >= SMOOTH_WINDOW:
        kernel   = np.ones(SMOOTH_WINDOW) / SMOOTH_WINDOW
        smoothed = np.convolve(step_losses, kernel, mode="valid")
        smooth_x = step_numbers[SMOOTH_WINDOW - 1:]

    # ── 1. Train vs Eval Loss ────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(step_numbers, step_losses, color="#2196F3", alpha=0.4, linewidth=0.8, label="Train loss (step)")
    if smoothed is not None:
        ax.plot(smooth_x, smoothed, color="#2196F3", linewidth=2, label=f"Train loss (smoothed w={SMOOTH_WINDOW})")
    if eval_steps:
        ax.plot(eval_steps, eval_losses, "s--", color="#F44336", linewidth=2, label="Eval loss")
    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("Train vs Eval Loss", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path1 = graphs_dir / "train_val_loss.png"
    fig.savefig(path1, dpi=150)
    plt.close(fig)
    print(f"  ✓ Saved: {path1}")

    # ── 2. Per-step raw loss ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(step_numbers, step_losses, color="#9C27B0", alpha=0.4, linewidth=0.8, label="Step loss")
    if smoothed is not None:
        ax.plot(smooth_x, smoothed, color="#9C27B0", linewidth=2, label=f"Smoothed (w={SMOOTH_WINDOW})")
    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("Per-Step Training Loss", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path2 = graphs_dir / "step_loss.png"
    fig.savefig(path2, dpi=150)
    plt.close(fig)
    print(f"  ✓ Saved: {path2}")

    # ── 3. Eval loss bar chart ───────────────────────────────────────────
    if eval_steps:
        best_idx = int(np.argmin(eval_losses))
        bar_colors = ["#4CAF50" if i == best_idx else "#90CAF9" for i in range(len(eval_losses))]
        fig, ax = plt.subplots(figsize=(max(6, len(eval_steps) * 1.0 + 2), 5))
        bars = ax.bar(range(len(eval_steps)), eval_losses, color=bar_colors, edgecolor="white")
        ax.set_xticks(range(len(eval_steps)))
        ax.set_xticklabels([f"step {s}" for s in eval_steps], rotation=30, ha="right")
        ax.set_xlabel("Checkpoint", fontsize=12)
        ax.set_ylabel("Eval Loss", fontsize=12)
        ax.set_title("Eval Loss per Checkpoint  (green = best)", fontsize=14, fontweight="bold")
        for bar, v in zip(bars, eval_losses):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        path3 = graphs_dir / "eval_loss_bars.png"
        fig.savefig(path3, dpi=150)
        plt.close(fig)
        print(f"  ✓ Saved: {path3}")
    else:
        print("  (no eval checkpoints yet — skipping eval_loss_bars.png)")
        best_idx, bar_colors = 0, []

    # ── 4. 2×2 Dashboard ────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Multilingual Whisper Fine-Tuning  [HuggingFace Trainer]\n"
                 "Yoruba · Hausa · Igbo · Pidgin",
                 fontsize=14, fontweight="bold")

    # top-left: train + eval vs step
    axes[0, 0].plot(step_numbers, step_losses, color="#2196F3", alpha=0.4, linewidth=0.7)
    if smoothed is not None:
        axes[0, 0].plot(smooth_x, smoothed, color="#2196F3", linewidth=2, label="Train (smoothed)")
    if eval_steps:
        axes[0, 0].plot(eval_steps, eval_losses, "s--", color="#F44336", linewidth=2, label="Eval")
    axes[0, 0].set_title("Train vs Eval Loss")
    axes[0, 0].set_xlabel("Step")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend(fontsize=9)
    axes[0, 0].grid(True, alpha=0.3)

    # top-right: eval loss bars
    if eval_steps:
        axes[0, 1].bar(range(len(eval_steps)), eval_losses, color=bar_colors, edgecolor="white")
        axes[0, 1].set_xticks(range(len(eval_steps)))
        axes[0, 1].set_xticklabels([f"s{s}" for s in eval_steps], rotation=30)
        axes[0, 1].set_title("Eval Loss per Checkpoint")
        axes[0, 1].set_xlabel("Checkpoint")
        axes[0, 1].set_ylabel("Eval Loss")
        axes[0, 1].grid(True, axis="y", alpha=0.3)
    else:
        axes[0, 1].text(0.5, 0.5, "No eval checkpoints",
                        ha="center", va="center", transform=axes[0, 1].transAxes, fontsize=11)
        axes[0, 1].axis("off")

    # bottom-left: raw + smoothed step loss
    axes[1, 0].plot(step_numbers, step_losses, color="#9C27B0", alpha=0.35, linewidth=0.7)
    if smoothed is not None:
        axes[1, 0].plot(smooth_x, smoothed, color="#9C27B0", linewidth=1.8)
    axes[1, 0].set_title("Step Loss (raw + smoothed)")
    axes[1, 0].set_xlabel("Step")
    axes[1, 0].set_ylabel("Loss")
    axes[1, 0].grid(True, alpha=0.3)

    # bottom-right: WER if available, else summary text
    wer_values = [w for w in eval_wers if w is not None]
    if wer_values and eval_steps:
        axes[1, 1].plot(eval_steps[:len(wer_values)], wer_values, "o-", color="#FF9800", linewidth=2)
        axes[1, 1].set_title("WER per Checkpoint")
        axes[1, 1].set_xlabel("Step")
        axes[1, 1].set_ylabel("WER (%)")
        axes[1, 1].grid(True, alpha=0.3)
    else:
        summary = [
            f"Final train loss : {step_losses[-1]:.4f}",
            f"Best eval loss   : {min(eval_losses):.4f}" if eval_losses else "Best eval loss   : N/A",
            f"Total steps      : {step_numbers[-1]}",
            f"Train log entries: {len(train_logs)}",
        ]
        for i, txt in enumerate(summary):
            axes[1, 1].text(0.05, 0.80 - i * 0.18, txt,
                            transform=axes[1, 1].transAxes, fontsize=10, fontfamily="monospace")
        axes[1, 1].set_title("Summary")
        axes[1, 1].axis("off")

    fig.tight_layout()
    path4 = graphs_dir / "dashboard.png"
    fig.savefig(path4, dpi=150)
    plt.close(fig)
    print(f"  ✓ Saved: {path4}")
    print(f"\n  All graphs saved to: {graphs_dir.absolute()}")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

print("\nStarting training...\n")
print("=" * 80)
trainer.train(resume_from_checkpoint=RESUME_FROM_CHECKPOINT or None)

# ---------------------------------------------------------------------------
# Save Final Model  (load_best_model_at_end=True means this IS the best model)
# ---------------------------------------------------------------------------

print("\n" + "=" * 80)
print("Training complete! Saving final model...")
trainer.save_model(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)

# Write a JSON record so Gradio (and humans) always know which checkpoint was best
import json
_best_ckpt = trainer.state.best_model_checkpoint   # e.g. "./whisper-small-nigerian/checkpoint-3000"
_best_step = int(_best_ckpt.split("-")[-1]) if _best_ckpt else None
_best_wer  = trainer.state.best_metric             # numeric WER value
best_info  = {
    "best_checkpoint": _best_ckpt,
    "best_step": _best_step,
    "best_metric_wer": round(float(_best_wer), 4) if _best_wer is not None else None,
    "final_model_dir": str(Path(OUTPUT_DIR).resolve()),
    "note": (
        "The model saved to 'final_model_dir' contains the best checkpoint weights. "
        "Load it with WhisperForConditionalGeneration.from_pretrained(final_model_dir)."
    ),
}
_info_path = Path(OUTPUT_DIR) / "best_model_info.json"
with open(_info_path, "w") as _f:
    json.dump(best_info, _f, indent=2)
print(f"\n✓ Best model info saved to: {_info_path}")
if _best_wer is not None:
    print(f"  Best checkpoint : {_best_ckpt}")
    print(f"  Best WER        : {_best_wer:.2f}%")

print("\nGenerating training graphs...")
save_training_graphs_hf(trainer, Path(OUTPUT_DIR))

print(f"\nModel saved to: {OUTPUT_DIR}")
print("\nTo use your model:")
print(f"from transformers import WhisperForConditionalGeneration, WhisperProcessor")
print(f'model = WhisperForConditionalGeneration.from_pretrained("{OUTPUT_DIR}")')
print(f'processor = WhisperProcessor.from_pretrained("{OUTPUT_DIR}")')
print("\nLanguages fine-tuned on: Yoruba, Hausa, Igbo, Pidgin English")
print(f"\nView training dashboard:")
print(f"  trackio show --project {TRACKIO_PROJECT}")

# Close the log file cleanly
_tee.close()
print(f"\nFull training log saved to: {_LOG_PATH.resolve()}")