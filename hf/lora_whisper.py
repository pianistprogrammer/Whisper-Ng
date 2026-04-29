import os
import sys
import gc
import unicodedata
import multiprocessing as mp
import torch
import numpy as np
import trackio
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import matplotlib
matplotlib.use("Agg")

from datasets import load_dataset, concatenate_datasets, DatasetDict, Audio
import pandas as pd
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    TrainerCallback,
    EarlyStoppingCallback,
    TrainerState,
    TrainerControl,
    TrainingArguments
)
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

if os.name == "nt":
    mp.freeze_support()

# ---------------------------------------------------------------------------
# Tee: mirror all stdout to a timestamped log file
# ---------------------------------------------------------------------------
class _Tee:
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
DATASET_NAME = "google/WaxalNLP"
TASK = "transcribe"

LANGUAGE_CONFIGS = [
    {"config": "yor_tts", "language": "yoruba"},
    {"config": "hau_tts", "language": "hausa"},
    {"config": "ibo_tts", "language": None},  # Igbo not natively supported, set language to None
    {"config": "pcm_tts", "language": "english"},  # Pidgin uses English tokenizer
]

# Additional local Common Voice datasets to merge into the finetuning pipeline
LOCAL_DATASETS = [
    {"dir": "datasets/cmn1qen4q00xjo107gln14ztz/cv-corpus-25.0-2026-03-09/ha", "language": "hausa"},
    {"dir": "datasets/cmn29vsoh019amm07d95id0mo/cv-corpus-25.0-2026-03-09/yo", "language": "yoruba"},
    {"dir": "datasets/cmn2cgr3101g2mm07mt1zagmz/cv-corpus-25.0-2026-03-09/pcm", "language": "english"},
    {"dir": "datasets/cmn2cp3yv01h6mm07x6tl0t1i/cv-corpus-25.0-2026-03-09/ig", "language": None},
]

RANDOM_SEED = 42
OUTPUT_DIR = "./whisper-small-nigerian-lora"
SAMPLE_RATE = 16000

# Training Hyperparameters for LoRA
BATCH_SIZE = 16
GRADIENT_ACCUMULATION_STEPS = 4  # Increased for larger effective batch size
LEARNING_RATE = 1e-4  # Lowered more to prevent extremely fast overfitting
WARMUP_STEPS = 50     # Scaled down to match frequent evals
MAX_STEPS = 1000      # 1000 steps is plenty (approx 125 epochs)
EVAL_STEPS = 40       # Evaluate frequently to catch the lowest validation loss
SAVE_STEPS = 40

IS_WINDOWS = os.name == "nt"
PREPROCESS_NUM_PROC = None if IS_WINDOWS else max(1, min(8, (os.cpu_count() or 1) - 1))

_RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
_LOG_PATH = Path(OUTPUT_DIR) / f"training_log_{_RUN_TIMESTAMP}.txt"
_tee = _Tee(_LOG_PATH)

print(f"Training log: {_LOG_PATH.resolve()}")
print(f"Run started : {_RUN_TIMESTAMP}")
print(f"Model       : {MODEL_NAME} with LoRA (PEFT 8-bit)")
print(f"Max steps   : {MAX_STEPS}  |  Eval every {EVAL_STEPS}  |  Save every {SAVE_STEPS}")
print(f"Batch size  : {BATCH_SIZE}  |  Grad accum: {GRADIENT_ACCUMULATION_STEPS}  "
      f"effective batch = {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
print(f"LR          : {LEARNING_RATE}  |  Warmup: {WARMUP_STEPS} steps")
print()

# ---------------------------------------------------------------------------
# Load Feature Extractor, Tokenizer and Processor
# ---------------------------------------------------------------------------
print("Loading feature extractor, tokenizer, and processor...")
feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL_NAME)
tokenizer = WhisperTokenizer.from_pretrained(MODEL_NAME, task=TASK)
processor = WhisperProcessor.from_pretrained(MODEL_NAME, task=TASK)

# ---------------------------------------------------------------------------
# Prepare Dataset
# ---------------------------------------------------------------------------
def prepare_dataset(batch, language):
    # Standardize the text to NFC format to handle mixed Yoruba diacritics
    clean_text = unicodedata.normalize("NFC", batch["text"])

    audio = batch["audio"]
    batch["input_features"] = feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"]
    ).input_features[0]

    if language is not None:
        tokenizer.set_prefix_tokens(language=language, task=TASK)
    else:
        # If language is None (like for Igbo), clear the language token
        tokenizer.set_prefix_tokens(language=None, task=TASK, predict_timestamps=False)

    # The actual column for text in the user's data seems to be "text"
    labels = tokenizer(
        clean_text, add_special_tokens=True, truncation=True, max_length=448
    ).input_ids

    batch["labels"] = labels
    return batch

print("Loading and preprocessing datasets for all 4 languages...")
all_train, all_val, all_test = [], [], []

# HF Datasets (WaxalNLP)
for lang_cfg in LANGUAGE_CONFIGS:
    cfg, lang = lang_cfg["config"], lang_cfg["language"]
    lang_str = str(lang).upper() if lang else "NONE"
    print(f"  Processing [HF] {lang_str} ({cfg})...", flush=True)
    
    ds = load_dataset(DATASET_NAME, cfg)
    ds = ds.cast_column("audio", Audio(sampling_rate=SAMPLE_RATE))
    
    ds = ds.map(
        prepare_dataset,
        fn_kwargs={"language": lang},
        remove_columns=ds["train"].column_names,
        num_proc=PREPROCESS_NUM_PROC,
    )
    
    all_train.append(ds["train"])
    all_val.append(ds["validation"])
    all_test.append(ds["test"])

# Local Common Voice Datasets
for lang_cfg in LOCAL_DATASETS:
    p = lang_cfg["dir"]
    lang = lang_cfg["language"]
    lang_str = str(lang).upper() if lang else "NONE"
    if not os.path.exists(p):
        continue
        
    print(f"  Processing [Local] {lang_str} ({p})...", flush=True)
    
    # Load and combine any existing train/test tsv files
    clips_dir = os.path.join(p, "clips")
    local_subsets = []
    
    for split in ["train", "test"]:
        tsv_path = os.path.join(p, f"{split}.tsv")
        if os.path.exists(tsv_path):
            ds_sub = load_dataset("csv", data_files=tsv_path, delimiter="\t", split="train")
            local_subsets.append(ds_sub)
            
    if not local_subsets:
        continue
        
    ds_local = concatenate_datasets(local_subsets)
    
    # Add audio path and text mapping
    def _add_audio_path(batch):
        batch["audio"] = os.path.join(clips_dir, str(batch["path"]))
        batch["text"] = str(batch["sentence"])
        return batch
        
    ds_local = ds_local.map(_add_audio_path)
    ds_local = ds_local.cast_column("audio", Audio(sampling_rate=SAMPLE_RATE))
    ds_local = ds_local.select_columns(["audio", "text", "sentence", "path"]) # clean up metadata
    
    ds_local = ds_local.map(
        prepare_dataset,
        fn_kwargs={"language": lang},
        remove_columns=ds_local.column_names,
        num_proc=PREPROCESS_NUM_PROC,
    )
    
    # Mathematical Dataset Resplit (70% Train, 15% Validation, 15% Test)
    # The percentages sum exactly to 100%
    # Step 1: Extract 70% for Train
    splits_1 = ds_local.train_test_split(train_size=0.70, seed=RANDOM_SEED)
    all_train.append(splits_1["train"])
    
    # Step 2: Split the remaining 30% into Validation (15/30 = 0.5) and Test (15/30 = 0.5)
    splits_2 = splits_1["test"].train_test_split(train_size=0.50, seed=RANDOM_SEED)
    all_val.append(splits_2["train"])
    all_test.append(splits_2["test"])

combined_train = concatenate_datasets(all_train).shuffle(seed=RANDOM_SEED)
combined_val   = concatenate_datasets(all_val).shuffle(seed=RANDOM_SEED)
combined_test  = concatenate_datasets(all_test).shuffle(seed=RANDOM_SEED)

common_voice = DatasetDict()
common_voice["train"] = concatenate_datasets([combined_train, combined_val]).shuffle(seed=RANDOM_SEED)
common_voice["test"]  = combined_test

del all_train, all_val, all_test, combined_train, combined_val, combined_test
gc.collect()

# ---------------------------------------------------------------------------
# Data Collator
# ---------------------------------------------------------------------------
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

# ---------------------------------------------------------------------------
# Load Model & Apply PEFT (LoRA)
# ---------------------------------------------------------------------------
print("Loading pre-trained Whisper model in 8-bit...")

from transformers import BitsAndBytesConfig
quantization_config = BitsAndBytesConfig(load_in_8bit=True)

model = WhisperForConditionalGeneration.from_pretrained(
    MODEL_NAME, 
    quantization_config=quantization_config, 
    device_map="auto"
)
model.config.use_cache = False
model.config.forced_decoder_ids = None # Explicitly remove forced language tokens globally
model.config.suppress_tokens = [] # Don't suppress tokens when learning a new language


# Prepare model for standard INT8 PEFT training
model = prepare_model_for_kbit_training(model)

def make_inputs_require_grad(module, input, output):
    output.requires_grad_(True)
model.model.encoder.conv1.register_forward_hook(make_inputs_require_grad)

print("Applying LoRA...")
config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"],
    lora_dropout=0.05,
    bias="none"
)

model = get_peft_model(model, config)
model.print_trainable_parameters()

# ---------------------------------------------------------------------------
# Save Callback
# ---------------------------------------------------------------------------
# This callback helps to save only the adapter weights and remove the base model weights.
class SavePeftModelCallback(TrainerCallback):
    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")
        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)

        pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        if os.path.exists(pytorch_model_path):
            os.remove(pytorch_model_path)
        return control

class StepLoggerCallback(TrainerCallback):
    """Write one structured line per logging/eval event to stdout (captured by _Tee)."""
    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        if not logs:
            return
            
        trackio.log(logs)
        
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

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    learning_rate=LEARNING_RATE,
    warmup_steps=WARMUP_STEPS,
    max_steps=MAX_STEPS,
    eval_strategy='steps',
    eval_steps=EVAL_STEPS,
    save_steps=SAVE_STEPS,
    fp16=True,  # Usually FP16 is used with 8-bit load
    per_device_eval_batch_size=BATCH_SIZE,
    logging_steps=25,
    remove_unused_columns=False,
    label_names=["labels"],
    predict_with_generate=False, # Cannot use generate during training loop with 8-bit PEFT
    load_best_model_at_end=True,     # <--- ADDED: Early Stopping 
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    save_total_limit=2,              # <--- Avoid saving 100s of checkpoints
)

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=common_voice["train"],
    eval_dataset=common_voice["test"], # Using combined test sets
    data_collator=data_collator,
    processing_class=processor.feature_extractor,
    callbacks=[SavePeftModelCallback, StepLoggerCallback(), EarlyStoppingCallback(early_stopping_patience=3)],
)

print("Starting training...")
trackio.init(
    project="whisper-ng-nigerian-lora",
    config={
        "model": MODEL_NAME,
        "dataset": DATASET_NAME,
        "batch_size": BATCH_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "learning_rate": LEARNING_RATE,
        "warmup_steps": WARMUP_STEPS,
        "max_steps": MAX_STEPS,
        "peft": "LoRA (8-bit)",
        "lora_r": 32,
        "lora_alpha": 64
    }
)
trainer.train()

print(f"Training complete! Model saved to {OUTPUT_DIR}")
model.save_pretrained(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)

print("\nDone! To evaluate, load the LoRA weights explicitly with PeftModel.from_pretrained()!")

trackio.finish()
_tee.close()
print(f"\nFull training log saved to: {_LOG_PATH.resolve()}")

