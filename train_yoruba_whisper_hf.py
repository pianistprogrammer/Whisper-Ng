"""
Fine-tuning OpenAI Whisper (base) on Yoruba (yor_tts) using Hugging Face Transformers.

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

from datasets import load_dataset, DatasetDict, Audio
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
DATASET_CONFIG = "yor_tts"
LANGUAGE = "yoruba"
TASK = "transcribe"

OUTPUT_DIR = "./whisper-base-yoruba"
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
tokenizer = WhisperTokenizer.from_pretrained(MODEL_NAME, language=LANGUAGE, task=TASK)
processor = WhisperProcessor.from_pretrained(MODEL_NAME, language=LANGUAGE, task=TASK)

# ---------------------------------------------------------------------------
# Load and Prepare Dataset
# ---------------------------------------------------------------------------

print("Loading dataset...")
raw_dataset = load_dataset(DATASET_NAME, DATASET_CONFIG)

# Cast audio to correct sampling rate FIRST
print(f"Setting audio sampling rate to {SAMPLE_RATE}Hz...")
raw_dataset = raw_dataset.cast_column("audio", Audio(sampling_rate=SAMPLE_RATE))

# Combine train and validation for training
print("Combining train and validation splits...")
common_voice = DatasetDict()
common_voice["train"] = load_dataset(DATASET_NAME, DATASET_CONFIG, split="train+validation")
common_voice["train"] = common_voice["train"].cast_column("audio", Audio(sampling_rate=SAMPLE_RATE))
common_voice["test"] = load_dataset(DATASET_NAME, DATASET_CONFIG, split="test")
common_voice["test"] = common_voice["test"].cast_column("audio", Audio(sampling_rate=SAMPLE_RATE))

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

# Set language and task for generation
model.generation_config.language = LANGUAGE
model.generation_config.task = TASK
model.generation_config.forced_decoder_ids = None

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
