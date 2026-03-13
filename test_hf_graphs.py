#!/usr/bin/env python3
"""
Quick HF-trainer graph test using 2 samples per language (8 total).
Runs for 6 steps, evaluates at steps 3 and 6, then generates all 4 charts.

Run:
    python test_hf_graphs.py
"""

import subprocess
import sys
from pathlib import Path

import torch
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Union

from datasets import load_dataset, concatenate_datasets, Audio
from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
import matplotlib
matplotlib.use("Agg")
import numpy as np

# ---------------------------------------------------------------------------
# Test config — tiny values so the run finishes in seconds
# ---------------------------------------------------------------------------

MODEL_NAME  = "openai/whisper-base"
DATASET_NAME = "google/WaxalNLP"
TASK        = "transcribe"
OUTPUT_DIR  = Path("test_hf_graph_output")
SAMPLE_RATE = 16000
SAMPLES_PER_LANG = 2

MAX_STEPS     = 6
EVAL_STEPS    = 3
LOGGING_STEPS = 1
SAVE_STEPS    = 1000   # avoid writing checkpoints to disk during the test
SMOOTH_WINDOW = 3      # smaller window for the short test run

LANGUAGE_CONFIGS = [
    {"config": "yor_tts", "language": "yoruba"},
    {"config": "hau_tts", "language": "hausa"},
    {"config": "ibo_tts", "language": "igbo"},
    {"config": "pcm_tts", "language": "english"},
]

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Load model + processor
# ---------------------------------------------------------------------------
print("=" * 70)
print("HF TRAINER GRAPH TEST  (2 per language × 4 languages, 6 steps)")
print("=" * 70)

print("\nLoading processor and model...", flush=True)
feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL_NAME)
tokenizer         = WhisperTokenizer.from_pretrained(MODEL_NAME, task=TASK)
processor         = WhisperProcessor.from_pretrained(MODEL_NAME, task=TASK)
model             = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
model.generation_config.task              = TASK
model.generation_config.forced_decoder_ids = None
model.generation_config.language          = None
print("  ✓ Loaded", flush=True)

# ---------------------------------------------------------------------------
# Load 2 samples per language
# ---------------------------------------------------------------------------
print("\nLoading 2 samples per language...", flush=True)
all_train, all_val = [], []
for lang_cfg in LANGUAGE_CONFIGS:
    cfg, lang = lang_cfg["config"], lang_cfg["language"]
    print(f"  {lang.upper()} ({cfg})", flush=True)
    ds = load_dataset(DATASET_NAME, cfg)
    ds = ds.cast_column("audio", Audio(sampling_rate=SAMPLE_RATE))
    all_train.append(ds["train"].select(range(SAMPLES_PER_LANG)))
    all_val.append(ds["validation"].select(range(SAMPLES_PER_LANG)))

combined_train = concatenate_datasets(all_train).shuffle(seed=42)
combined_val   = concatenate_datasets(all_val).shuffle(seed=42)
print(f"\n  Combined  train={len(combined_train)}  val={len(combined_val)}", flush=True)

# ---------------------------------------------------------------------------
# Preprocess
# ---------------------------------------------------------------------------
def prepare_dataset(batch):
    audio = batch["audio"]
    batch["input_features"] = feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"]
    ).input_features[0]
    labels = tokenizer(batch["text"], add_special_tokens=True).input_ids[:448]
    batch["labels"] = labels
    return batch

print("\nPreprocessing...", flush=True)
train_ds = combined_train.map(prepare_dataset, remove_columns=combined_train.column_names)
val_ds   = combined_val.map(prepare_dataset,   remove_columns=combined_val.column_names)
print(f"  ✓ train={len(train_ds)}  val={len(val_ds)}", flush=True)

# ---------------------------------------------------------------------------
# Data Collator
# ---------------------------------------------------------------------------
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        decoder_attention_mask = labels_batch.attention_mask
        if (labels_batch["input_ids"][:, 0] == self.decoder_start_token_id).all().cpu().item():
            decoder_attention_mask = decoder_attention_mask[:, 1:]
        batch["decoder_attention_mask"] = decoder_attention_mask
        return batch

# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
training_args = Seq2SeqTrainingArguments(
    output_dir=str(OUTPUT_DIR),
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    learning_rate=1e-5,
    warmup_steps=0,
    max_steps=MAX_STEPS,
    gradient_checkpointing=False,
    fp16=torch.cuda.is_available(),
    eval_strategy="steps",
    eval_steps=EVAL_STEPS,
    save_steps=SAVE_STEPS,
    logging_steps=LOGGING_STEPS,
    predict_with_generate=True,
    generation_max_length=225,
    report_to=[],    # disable tensorboard for the test
    load_best_model_at_end=False,
    push_to_hub=False,
)

data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id,
)

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=data_collator,
)

print(f"\nTraining for {MAX_STEPS} steps (eval every {EVAL_STEPS})...", flush=True)
trainer.train()
print("  ✓ Training done", flush=True)

# ---------------------------------------------------------------------------
# Generate graphs (function inlined — avoids importing the full training script
# which has top-level execution code)
# ---------------------------------------------------------------------------
import matplotlib.pyplot as plt

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

    step_numbers = [e["step"]      for e in train_logs]
    step_losses  = [e["loss"]      for e in train_logs]
    eval_steps   = [e["step"]      for e in eval_logs]
    eval_losses  = [e["eval_loss"] for e in eval_logs]
    eval_wers    = [e.get("eval_wer") for e in eval_logs]

    smoothed = smooth_x = None
    if len(step_losses) >= SMOOTH_WINDOW:
        kernel   = np.ones(SMOOTH_WINDOW) / SMOOTH_WINDOW
        smoothed = np.convolve(step_losses, kernel, mode="valid")
        smooth_x = step_numbers[SMOOTH_WINDOW - 1:]

    # ── 1. Train vs Eval Loss ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(step_numbers, step_losses, color="#2196F3", alpha=0.4, linewidth=0.8, label="Train loss (step)")
    if smoothed is not None:
        ax.plot(smooth_x, smoothed, color="#2196F3", linewidth=2, label=f"Train loss (smoothed w={SMOOTH_WINDOW})")
    if eval_steps:
        ax.plot(eval_steps, eval_losses, "s--", color="#F44336", linewidth=2, label="Eval loss")
    ax.set(xlabel="Step", ylabel="Loss", title="Train vs Eval Loss")
    ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(graphs_dir / "train_val_loss.png", dpi=150); plt.close(fig)
    print(f"  ✓ Saved: {graphs_dir / 'train_val_loss.png'}")

    # ── 2. Per-step raw loss ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(step_numbers, step_losses, color="#9C27B0", alpha=0.4, linewidth=0.8, label="Step loss")
    if smoothed is not None:
        ax.plot(smooth_x, smoothed, color="#9C27B0", linewidth=2, label=f"Smoothed (w={SMOOTH_WINDOW})")
    ax.set(xlabel="Step", ylabel="Loss", title="Per-Step Training Loss")
    ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(graphs_dir / "step_loss.png", dpi=150); plt.close(fig)
    print(f"  ✓ Saved: {graphs_dir / 'step_loss.png'}")

    # ── 3. Eval loss bar chart ──────────────────────────────────────────────
    if eval_steps:
        best_idx   = int(np.argmin(eval_losses))
        bar_colors = ["#4CAF50" if i == best_idx else "#90CAF9" for i in range(len(eval_losses))]
        fig, ax = plt.subplots(figsize=(max(6, len(eval_steps) + 2), 5))
        bars = ax.bar(range(len(eval_steps)), eval_losses, color=bar_colors, edgecolor="white")
        ax.set_xticks(range(len(eval_steps)))
        ax.set_xticklabels([f"step {s}" for s in eval_steps], rotation=30, ha="right")
        ax.set(xlabel="Checkpoint", ylabel="Eval Loss",
               title="Eval Loss per Checkpoint  (green = best)")
        for bar, v in zip(bars, eval_losses):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout(); fig.savefig(graphs_dir / "eval_loss_bars.png", dpi=150); plt.close(fig)
        print(f"  ✓ Saved: {graphs_dir / 'eval_loss_bars.png'}")
    else:
        print("  (no eval checkpoints — skipping eval_loss_bars.png)")
        best_idx, bar_colors = 0, []

    # ── 4. 2×2 Dashboard ───────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Multilingual Whisper Fine-Tuning  [HuggingFace Trainer]\n"
                 "Yoruba · Hausa · Igbo · Pidgin", fontsize=14, fontweight="bold")

    axes[0, 0].plot(step_numbers, step_losses, color="#2196F3", alpha=0.4, linewidth=0.7)
    if smoothed is not None:
        axes[0, 0].plot(smooth_x, smoothed, color="#2196F3", linewidth=2, label="Train")
    if eval_steps:
        axes[0, 0].plot(eval_steps, eval_losses, "s--", color="#F44336", linewidth=2, label="Eval")
    axes[0, 0].set_title("Train vs Eval Loss"); axes[0, 0].legend(fontsize=9); axes[0, 0].grid(True, alpha=0.3)

    if eval_steps:
        axes[0, 1].bar(range(len(eval_steps)), eval_losses, color=bar_colors, edgecolor="white")
        axes[0, 1].set_xticks(range(len(eval_steps)))
        axes[0, 1].set_xticklabels([f"s{s}" for s in eval_steps], rotation=30)
        axes[0, 1].set_title("Eval Loss per Checkpoint"); axes[0, 1].grid(True, axis="y", alpha=0.3)
    else:
        axes[0, 1].text(0.5, 0.5, "No eval checkpoints", ha="center", va="center",
                        transform=axes[0, 1].transAxes); axes[0, 1].axis("off")

    axes[1, 0].plot(step_numbers, step_losses, color="#9C27B0", alpha=0.35, linewidth=0.7)
    if smoothed is not None:
        axes[1, 0].plot(smooth_x, smoothed, color="#9C27B0", linewidth=1.8)
    axes[1, 0].set_title("Step Loss (raw + smoothed)"); axes[1, 0].grid(True, alpha=0.3)

    wer_values = [w for w in eval_wers if w is not None]
    if wer_values and eval_steps:
        axes[1, 1].plot(eval_steps[:len(wer_values)], wer_values, "o-", color="#FF9800", linewidth=2)
        axes[1, 1].set_title("WER per Checkpoint"); axes[1, 1].grid(True, alpha=0.3)
    else:
        summary = [
            f"Final train loss : {step_losses[-1]:.4f}",
            f"Best eval loss   : {min(eval_losses):.4f}" if eval_losses else "Best eval loss   : N/A",
            f"Total steps      : {step_numbers[-1]}",
        ]
        for i, txt in enumerate(summary):
            axes[1, 1].text(0.05, 0.80 - i * 0.18, txt,
                            transform=axes[1, 1].transAxes, fontsize=10, fontfamily="monospace")
        axes[1, 1].set_title("Summary"); axes[1, 1].axis("off")

    fig.tight_layout()
    fig.savefig(graphs_dir / "dashboard.png", dpi=150); plt.close(fig)
    print(f"  ✓ Saved: {graphs_dir / 'dashboard.png'}")
    print(f"\n  All graphs saved to: {graphs_dir.absolute()}")


print("\nGenerating graphs...", flush=True)
save_training_graphs_hf(trainer, OUTPUT_DIR)

# ---------------------------------------------------------------------------
# Open the newest graphs_* dir in Finder
# ---------------------------------------------------------------------------
graph_dirs = sorted(OUTPUT_DIR.glob("graphs_*"))
if graph_dirs:
    newest = graph_dirs[-1]
    print(f"\nOpening {newest} in Finder...")
    subprocess.run(["open", str(newest)])

print("\n✓ Test complete!")
