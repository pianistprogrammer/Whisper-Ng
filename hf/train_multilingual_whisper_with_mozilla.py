"""
Fine-tuning OpenAI Whisper (small) on Nigerian languages
combining HuggingFace and Mozilla Data Collective datasets.

Languages:
  - Yoruba (HF: yor_tts, Mozilla: common-voice-yoruba)
  - Hausa (HF: hau_tts, Mozilla: common-voice-hausa)
  - Igbo (HF: ibo_tts, Mozilla: common-voice-igbo)
  - Pidgin English (HF: pcm_tts)

Requirements:
    pip install transformers datasets accelerate evaluate jiwer tensorboard python-dotenv tqdm

Run:
    python train_multilingual_whisper_with_mozilla.py

Outputs (all saved to OUTPUT_DIR):
    - Model checkpoints
    - Training logs
    - Evaluation metrics
"""

import torch
import sys
import os
import gc
from pathlib import Path

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Add parent directory to path to import mozilla_dataset_loader
sys.path.insert(0, str(Path(__file__).parent.parent))
from mozilla_dataset_loader import load_mozilla_datasets

# Apple Silicon: MPS unified memory is managed by the OS
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
from typing import Any, Dict, List, Union
import evaluate
import re
import matplotlib
matplotlib.use("Agg")
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
# Tee: mirror all stdout to a timestamped log file
# ---------------------------------------------------------------------------

class _Tee:
    """Write every print() call to both the terminal and a log file."""
    def __init__(self, log_path: Path):
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(log_path, "w", buffering=1, encoding="utf-8")
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
TASK = "transcribe"

# HuggingFace datasets
HF_DATASET_NAME = "google/WaxalNLP"
HF_LANGUAGE_CONFIGS = [
    {"config": "yor_tts", "language": "yoruba"},
    {"config": "hau_tts", "language": "hausa"},
    {"config": "ibo_tts", "language": None},      # Igbo not in Whisper vocab
    {"config": "pcm_tts", "language": "english"},  # Pidgin uses English tokenizer
]

# Mozilla Data Collective datasets
# All verified as audio datasets (ASR/TTS tasks)
# Total: 7 datasets, ~1.67 GB
MOZILLA_DATASET_IDS = [
    "cmnopto3q00t0mf07v2dtc0ej",  # Hausa-TTS-Dataset (270 MB)
    "cmn1qen4q00xjo107gln14ztz",  # Common Voice 25.0 - Hausa (250 MB)
    "cmo1nlaah0071mk077mw0qhpv",  # Yoruba-TTS-Dataset (310 MB)
    "cmn29vsoh019amm07d95id0mo",  # Common Voice 25.0 - Yoruba (160 MB)
    "cmn2cp3yv01h6mm07x6tl0t1i",  # Common Voice 25.0 - Igbo (360 MB)
    "cmnykldcz010knu0737o8bgh9",  # Naija-TTS-Dataset / Pidgin (320 MB)
    "cmn2cgr3101g2mm07mt1zagmz",  # Common Voice 25.0 - Nigerian Pidgin (280 MB)
]

# Language mapping for Mozilla datasets (auto-detect from locale/name)
MOZILLA_LANGUAGE_MAP = {
    "yoruba": "yoruba",
    "yor": "yoruba",
    "yo": "yoruba",
    "hausa": "hausa",
    "hau": "hausa",
    "ha": "hausa",
    "igbo": None,  # Not in Whisper vocab
    "ig": None,
    "pidgin": "english",  # Pidgin uses English tokenizer
    "naija": "english",
    "pcm": "english",
}

RANDOM_SEED = 42
OUTPUT_DIR = "./whisper-small-nigerian-mozilla"
SAMPLE_RATE = 16000
MOZILLA_CACHE_DIR = "./datasets"

# Training hyperparameters
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 1e-5
WARMUP_STEPS = 500
MAX_STEPS = 4000
EVAL_STEPS = 1000
SAVE_STEPS = 1000
LOGGING_STEPS = 25
SMOOTH_WINDOW = 10

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
# Prepare Data
# ---------------------------------------------------------------------------

def prepare_dataset(batch, language):
    """Prepare a single batch for training, using the explicit language."""
    audio = batch["audio"]

    # Compute log-Mel input features
    batch["input_features"] = feature_extractor(
        audio["array"],
        sampling_rate=audio["sampling_rate"]
    ).input_features[0]

    # Set language token for tokenizer
    if language is not None:
        tokenizer.set_prefix_tokens(language=language, task=TASK)
    else:
        tokenizer.set_prefix_tokens(task=TASK)

    labels = tokenizer(
        batch["text"], add_special_tokens=True, truncation=True, max_length=448
    ).input_ids

    batch["labels"] = labels
    return batch

# ---------------------------------------------------------------------------
# Load and Prepare Datasets
# ---------------------------------------------------------------------------

print("\n" + "="*80)
print("LOADING HUGGINGFACE DATASETS")
print("="*80)

all_train, all_val, all_test = [], [], []

# Load HuggingFace datasets
for lang_cfg in HF_LANGUAGE_CONFIGS:
    cfg, lang = lang_cfg["config"], lang_cfg["language"]
    print(f"\nProcessing HF {(lang or cfg).upper()} ({cfg})...", flush=True)

    ds = load_dataset(HF_DATASET_NAME, cfg)
    ds = ds.cast_column("audio", Audio(sampling_rate=SAMPLE_RATE))
    print(f"  train={len(ds['train']):,}  val={len(ds['validation']):,}  test={len(ds['test']):,}")

    col_names = ds["train"].column_names

    # Map each split
    for split in ["train", "validation", "test"]:
        mapped = ds[split].map(
            prepare_dataset,
            fn_kwargs={"language": lang},
            remove_columns=col_names,
        )
        if split == "train":
            all_train.append(mapped)
        elif split == "validation":
            all_val.append(mapped)
        else:
            all_test.append(mapped)

    del ds
    gc.collect()

# Load Mozilla datasets
if MOZILLA_DATASET_IDS:
    print("\n" + "="*80)
    print("LOADING MOZILLA DATA COLLECTIVE DATASETS")
    print("="*80)

    client_id = os.getenv("client_id")
    api_key = os.getenv("api_key")

    if not client_id or not api_key:
        print("WARNING: Mozilla credentials not found in .env file")
        print("Skipping Mozilla datasets...")
    else:
        try:
            mozilla_datasets = load_mozilla_datasets(
                dataset_ids=MOZILLA_DATASET_IDS,
                client_id=client_id,
                api_key=api_key,
                cache_dir=MOZILLA_CACHE_DIR,
                sample_rate=SAMPLE_RATE
            )

            # Process each Mozilla dataset
            for dataset_id, ds_dict in mozilla_datasets.items():
                # Infer language from dataset ID
                lang = None
                for lang_name, whisper_lang in MOZILLA_LANGUAGE_MAP.items():
                    if lang_name.lower() in dataset_id.lower():
                        lang = whisper_lang
                        break

                print(f"\nProcessing Mozilla {dataset_id} (language: {lang or 'unknown'})...")

                # Map each split
                for split in ds_dict:
                    col_names = ds_dict[split].column_names
                    mapped = ds_dict[split].map(
                        prepare_dataset,
                        fn_kwargs={"language": lang},
                        remove_columns=col_names,
                    )

                    if split == "train":
                        all_train.append(mapped)
                        print(f"  Added {len(mapped):,} training examples")
                    elif split == "validation":
                        all_val.append(mapped)
                        print(f"  Added {len(mapped):,} validation examples")
                    elif split == "test":
                        all_test.append(mapped)
                        print(f"  Added {len(mapped):,} test examples")

                del ds_dict
                gc.collect()

        except Exception as e:
            print(f"ERROR loading Mozilla datasets: {e}")
            print("Continuing with HuggingFace datasets only...")

# Combine all datasets
print("\n" + "="*80)
print("COMBINING ALL DATASETS")
print("="*80)

common_voice = DatasetDict()
common_voice["train"] = concatenate_datasets(all_train + all_val).shuffle(seed=RANDOM_SEED)
common_voice["test"] = concatenate_datasets(all_test).shuffle(seed=RANDOM_SEED)

print(f"  Combined train : {len(common_voice['train']):,}")
print(f"  Combined test  : {len(common_voice['test']):,}")

# Free memory
del all_train, all_val, all_test
gc.collect()

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
        # Pad input features
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # Pad labels
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # Replace padding with -100 for loss computation
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # Cut bos token if present
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        # Set decoder attention mask
        decoder_attention_mask = labels_batch.attention_mask
        if (labels_batch["input_ids"][:, 0] == self.decoder_start_token_id).all().cpu().item():
            decoder_attention_mask = decoder_attention_mask[:, 1:]
        batch["decoder_attention_mask"] = decoder_attention_mask

        return batch

# ---------------------------------------------------------------------------
# Load Pre-trained Model
# ---------------------------------------------------------------------------

print("\nLoading pre-trained Whisper model...")
model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
model.config.use_cache = False
model = model.to(DEVICE)

if DEVICE == "cuda":
    model.gradient_checkpointing_enable()

# Multilingual mode
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
    from jiwer import wer as compute_wer

    class WERMetric:
        def compute(self, predictions, references):
            return compute_wer(references, predictions)

    metric = WERMetric()

def normalize_text(text):
    """Lowercase and remove all punctuation for fair WER comparison."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()

def compute_metrics(pred):
    """Compute WER metric with normalized text."""
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # Replace -100 with pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    # Normalize
    pred_str = [normalize_text(p) for p in pred_str]
    label_str = [normalize_text(l) for l in label_str]

    # Filter empty references
    valid_pairs = [(p, l) for p, l in zip(pred_str, label_str) if l.strip()]
    if not valid_pairs:
        return {"wer": 100.0}
    valid_preds, valid_labels = zip(*valid_pairs)

    wer = 100 * metric.compute(predictions=list(valid_preds), references=list(valid_labels))

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
    gradient_checkpointing=DEVICE == "cuda",
    fp16=DEVICE == "cuda",
    bf16=False,
    eval_strategy="steps",
    per_device_eval_batch_size=1,
    predict_with_generate=True,
    generation_max_length=128,
    save_steps=SAVE_STEPS,
    eval_steps=EVAL_STEPS,
    logging_steps=LOGGING_STEPS,
    report_to=["tensorboard"],
    run_name=f"whisper-small-nigerian-mozilla-{_RUN_TIMESTAMP}",
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    save_total_limit=2,
    push_to_hub=False,
    dataloader_pin_memory=False,
)

# ---------------------------------------------------------------------------
# Initialize Trainer
# ---------------------------------------------------------------------------

from transformers import TrainerCallback

class StepLoggerCallback(TrainerCallback):
    """Write one structured line per logging/eval event."""
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
    callbacks=[StepLoggerCallback()],
)

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

print("\nStarting training...\n")
print("=" * 80)
trainer.train(resume_from_checkpoint=RESUME_FROM_CHECKPOINT or None)

# ---------------------------------------------------------------------------
# Save Final Model
# ---------------------------------------------------------------------------

print("\n" + "=" * 80)
print("Training complete! Saving final model...")
trainer.save_model(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)

# Save best model info
import json
_best_ckpt = trainer.state.best_model_checkpoint
_best_step = int(_best_ckpt.split("-")[-1]) if _best_ckpt else None
_best_wer = trainer.state.best_metric
best_info = {
    "best_checkpoint": _best_ckpt,
    "best_step": _best_step,
    "best_metric_wer": round(float(_best_wer), 4) if _best_wer is not None else None,
    "final_model_dir": str(Path(OUTPUT_DIR).resolve()),
    "note": "The model in final_model_dir contains the best checkpoint weights.",
}
_info_path = Path(OUTPUT_DIR) / "best_model_info.json"
with open(_info_path, "w") as _f:
    json.dump(best_info, _f, indent=2)

print(f"\n✓ Best model info saved to: {_info_path}")
if _best_wer is not None:
    print(f"  Best checkpoint : {_best_ckpt}")
    print(f"  Best WER        : {_best_wer:.2f}%")

print(f"\nModel saved to: {OUTPUT_DIR}")
print("\nTo use your model:")
print(f'  from transformers import WhisperForConditionalGeneration, WhisperProcessor')
print(f'  model = WhisperForConditionalGeneration.from_pretrained("{OUTPUT_DIR}")')
print(f'  processor = WhisperProcessor.from_pretrained("{OUTPUT_DIR}")')
print("\nLanguages fine-tuned: Yoruba, Hausa, Igbo, Pidgin English")
print(f"  - HuggingFace datasets: {len(HF_LANGUAGE_CONFIGS)} languages")
print(f"  - Mozilla datasets: {len(MOZILLA_DATASET_IDS)} datasets")

# Close log file
_tee.close()
print(f"\nFull training log: {_LOG_PATH.resolve()}")
