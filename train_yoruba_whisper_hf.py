"""
Fine-tuning OpenAI Whisper (base) on 4 Nigerian languages using Hugging Face Transformers.

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
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import evaluate

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
# Config
# ---------------------------------------------------------------------------

MODEL_NAME = "openai/whisper-base"
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
OUTPUT_DIR = "./whisper-base-nigerian"
SAMPLE_RATE = 16000

# Training hyperparameters
BATCH_SIZE = 16
GRADIENT_ACCUMULATION_STEPS = 1
LEARNING_RATE = 1e-5
WARMUP_STEPS = 500
MAX_STEPS = 5000
EVAL_STEPS = 1000
SAVE_STEPS = 1000
LOGGING_STEPS = 25

# ---------------------------------------------------------------------------
# Load Feature Extractor, Tokenizer and Processor
# ---------------------------------------------------------------------------

print("Loading feature extractor, tokenizer, and processor...")
feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL_NAME)
# No language forced — multilingual mode lets Whisper handle all 4 languages
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
    num_proc=4
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

# Multilingual mode — do NOT force a single language token
model.generation_config.task = TASK
model.generation_config.forced_decoder_ids = None
model.generation_config.language = None

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
    gradient_checkpointing=False,  # Disabled due to compatibility issues
    fp16=torch.cuda.is_available(),  # Use fp16 only if CUDA is available
    eval_strategy="steps",
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=SAVE_STEPS,
    eval_steps=EVAL_STEPS,
    logging_steps=LOGGING_STEPS,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False,  # Set to True if you want to push to Hub
)

# ---------------------------------------------------------------------------
# Initialize Trainer
# ---------------------------------------------------------------------------

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
)

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

print("\nStarting training...\n")
print("=" * 80)
trainer.train()

# ---------------------------------------------------------------------------
# Save Final Model
# ---------------------------------------------------------------------------

print("\n" + "=" * 80)
print("Training complete! Saving final model...")
trainer.save_model(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)

print(f"\nModel saved to: {OUTPUT_DIR}")
print("\nTo use your model:")
print(f"from transformers import WhisperForConditionalGeneration, WhisperProcessor")
print(f'model = WhisperForConditionalGeneration.from_pretrained("{OUTPUT_DIR}")')
print(f'processor = WhisperProcessor.from_pretrained("{OUTPUT_DIR}")')
print("\nLanguages fine-tuned on: Yoruba, Hausa, Igbo, Pidgin English")