import os
import gc
import re
import torch
import unicodedata
import evaluate
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import WhisperProcessor, WhisperForConditionalGeneration, BitsAndBytesConfig
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
from peft import PeftModel, PeftConfig
from datasets import load_dataset, Audio, concatenate_datasets
from dataclasses import dataclass
from typing import Any, Dict, List, Union

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PEFT_MODEL_ID = "./whisper-small-nigerian-lora"
DATASET_NAME = "google/WaxalNLP"
TASK = "transcribe"
SAMPLE_RATE = 16000
BATCH_SIZE = 16

LANGUAGE_CONFIGS = [
    {"config": "yor_tts", "language": "yoruba", "local_dir": "datasets/cmn29vsoh019amm07d95id0mo/cv-corpus-25.0-2026-03-09/yo"},
    {"config": "hau_tts", "language": "hausa", "local_dir": "datasets/cmn1qen4q00xjo107gln14ztz/cv-corpus-25.0-2026-03-09/ha"},
    {"config": "ibo_tts", "language": None, "local_dir": "datasets/cmn2cp3yv01h6mm07x6tl0t1i/cv-corpus-25.0-2026-03-09/ig"},  # Igbo unset as in training
    {"config": "pcm_tts", "language": "english", "local_dir": "datasets/cmn2cgr3101g2mm07mt1zagmz/cv-corpus-25.0-2026-03-09/pcm"},
]

# ---------------------------------------------------------------------------
# Load Model, Processor, and PEFT Adapters
# ---------------------------------------------------------------------------
print(f"Loading PEFT config from {PEFT_MODEL_ID}...")
peft_config = PeftConfig.from_pretrained(PEFT_MODEL_ID)
processor = WhisperProcessor.from_pretrained(peft_config.base_model_name_or_path, task=TASK)

print("Loading base model in 8-bit...")
quantization_config = BitsAndBytesConfig(load_in_8bit=True)
model = WhisperForConditionalGeneration.from_pretrained(
    peft_config.base_model_name_or_path, 
    quantization_config=quantization_config, 
    device_map="auto"
)

print("Attaching LoRA adapters...")
model = PeftModel.from_pretrained(model, PEFT_MODEL_ID)
model.config.use_cache = True

# Clear out forced decoders natively to allow per-language overrides
model.generation_config.forced_decoder_ids = None 
model.generation_config.suppress_tokens = []
if hasattr(model.config, "forced_decoder_ids"):
    delattr(model.config, "forced_decoder_ids")
if hasattr(model.config, "suppress_tokens"):
    delattr(model.config, "suppress_tokens")

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
# Evaluation Loop (Per Language)
# ---------------------------------------------------------------------------
# Use the basic normalizer which handles unicode better than a raw regex
# remove_diacritics=False ensures it doesn't strip Yoruba marks or other tonal info
normalizer = BasicTextNormalizer(remove_diacritics=False)

def normalize_text(text):
    # Standardize to NFC first, then lowercase
    clean_text = unicodedata.normalize("NFC", text).lower()
    # Strip punctuation but keep word characters and whitespace
    clean_text = re.sub(r'[^\w\s]', '', clean_text)
    # Collapse multiple spaces
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
    return clean_text

metric = evaluate.load("wer")
all_predictions = []
all_references = []
all_normalized_predictions = []
all_normalized_references = []

model.eval()

for lang_cfg in LANGUAGE_CONFIGS:
    cfg = lang_cfg["config"]
    lang = lang_cfg["language"]
    lang_str = str(lang).upper() if lang else "NONE"
    
    print(f"\nEvaluating: {lang_str} ({cfg})")
    
    # 1. Load HF Test Split
    ds_hf = load_dataset(DATASET_NAME, cfg, split="test").cast_column("audio", Audio(sampling_rate=SAMPLE_RATE))
    # Ensure they have uniform schema to avoid concatenation issues
    ds_hf = ds_hf.select_columns(["audio", "text"])
    
    datasets_to_concat = [ds_hf]
    
    # 2. Load Local Common Voice Test Split
    local_dir = lang_cfg.get("local_dir")
    if local_dir and os.path.exists(local_dir):
        tsv_path = os.path.join(local_dir, "test.tsv")
        if os.path.exists(tsv_path):
            clips_dir = os.path.join(local_dir, "clips")
            ds_local = load_dataset("csv", data_files=tsv_path, delimiter="\t", split="train") # Loads the file itself
            
            def _add_audio_path(batch):
                batch["audio"] = os.path.join(clips_dir, str(batch["path"]))
                batch["text"] = str(batch["sentence"])
                return batch
            
            ds_local = ds_local.map(_add_audio_path)
            ds_local = ds_local.cast_column("audio", Audio(sampling_rate=SAMPLE_RATE))
            ds_local = ds_local.select_columns(["audio", "text"])
            
            datasets_to_concat.append(ds_local)
            
    ds = concatenate_datasets(datasets_to_concat)
    
    # Configure processor tokenizer for correct label generation
    if lang is not None:
        processor.tokenizer.set_prefix_tokens(language=lang, task=TASK)
        forced_decoder_ids = processor.get_decoder_prompt_ids(language=lang, task=TASK)
    else:
        processor.tokenizer.set_prefix_tokens(language=None, task=TASK, predict_timestamps=False)
        forced_decoder_ids = processor.get_decoder_prompt_ids(language=None, task=TASK)
        if hasattr(processor.tokenizer, "prefix_tokens"):
             forced_decoder_ids = [[i, token_id] for i, token_id in enumerate(processor.tokenizer.prefix_tokens)]
             
    def prepare_dataset(batch):
        clean_text = unicodedata.normalize("NFC", batch["text"])
        audio = batch["audio"]
        batch["input_features"] = processor.feature_extractor(
            audio["array"], sampling_rate=audio["sampling_rate"]
        ).input_features[0]
        batch["labels"] = processor.tokenizer(clean_text).input_ids
        return batch

    ds = ds.map(prepare_dataset, remove_columns=ds.column_names, num_proc=1 if os.name == "nt" else 4)
    eval_dataloader = DataLoader(ds, batch_size=BATCH_SIZE, collate_fn=data_collator)

    predictions = []
    references = []
    
    for step, batch in enumerate(tqdm(eval_dataloader, desc=f"Evaluating {lang_str}")):
        with torch.amp.autocast('cuda'):
            with torch.no_grad():
                generated_tokens = model.generate(
                    input_features=batch["input_features"].to("cuda"),
                    max_new_tokens=255,
                    forced_decoder_ids=forced_decoder_ids
                ).cpu().numpy()
                
                labels = batch["labels"].cpu().numpy()
                labels = np.where(labels != -100, labels, processor.tokenizer.pad_token_id)
                
                decoded_preds = processor.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                decoded_labels = processor.tokenizer.batch_decode(labels, skip_special_tokens=True)
                
                if step == 0:
                    print(f"\n[SAMPLE] REF : {decoded_labels[0]}\n[SAMPLE] PRED: {decoded_preds[0]}\n")
                
                predictions.extend(decoded_preds)
                references.extend(decoded_labels)

        del generated_tokens, labels, batch
        gc.collect()
        
    normalized_preds = [normalize_text(pred) for pred in predictions]
    normalized_refs = [normalize_text(label) for label in references]
    
    # Filter empty references to avoid division by zero
    valid_preds, valid_refs, valid_norm_preds, valid_norm_refs = [], [], [], []
    for p, r, npred, nref in zip(predictions, references, normalized_preds, normalized_refs):
        if len(r.strip()) > 0:
            valid_preds.append(p)
            valid_refs.append(r)
        if len(nref.strip()) > 0:
            valid_norm_preds.append(npred)
            valid_norm_refs.append(nref)
            
    wer = 100 * metric.compute(predictions=valid_preds, references=valid_refs) if valid_refs else 0
    norm_wer = 100 * metric.compute(predictions=valid_norm_preds, references=valid_norm_refs) if valid_norm_refs else 0
    
    print(f"--> {lang_str} WER: {wer:.2f}% | Normalized WER: {norm_wer:.2f}%")
    
    all_predictions.extend(valid_preds)
    all_references.extend(valid_refs)
    all_normalized_predictions.extend(valid_norm_preds)
    all_normalized_references.extend(valid_norm_refs)

final_wer = 100 * metric.compute(predictions=all_predictions, references=all_references)
final_norm_wer = 100 * metric.compute(predictions=all_normalized_predictions, references=all_normalized_references)

print("\n================================================================================")
print(f"Final Word Error Rate (WER)            : {final_wer:.2f}%")
print(f"Final Normalized WER (Lower is better) : {final_norm_wer:.2f}%")
print("================================================================================")
