#!/usr/bin/env python3
"""
Evaluate the Whisper LoRA fine-tune on Yoruba, Hausa, Igbo, and Nigerian Pidgin.

Saves per-sample reference/hypothesis pairs to JSON files consumable by
generate_test_dashboard.py:
    test_results_yor.json
    test_results_hau.json
    test_results_ibo.json
    test_results_pcm.json

Usage:
    python lora/eval_lora_whisper.py
    python lora/eval_lora_whisper.py --model ./whisper-small-nigerian-lora
    python lora/eval_lora_whisper.py --langs yor hau  # subset of languages
    python lora/eval_lora_whisper.py --out-dir results/
"""

import argparse
import gc
import json
import os
import re
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Union

import evaluate
import numpy as np
import torch
from datasets import Audio, concatenate_datasets, load_dataset
from peft import PeftConfig, PeftModel
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    BitsAndBytesConfig,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)
from transformers.models.whisper.english_normalizer import BasicTextNormalizer

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_PEFT_MODEL_ID = "./whisper-small-nigerian-lora"
DATASET_NAME = "google/WaxalNLP"
TASK = "transcribe"
SAMPLE_RATE = 16_000
BATCH_SIZE = 16

LANGUAGE_CONFIGS = [
    {
        "key": "yor",
        "config": "yor_tts",
        "language": "yoruba",
        "local_dir": "datasets/cmn29vsoh019amm07d95id0mo/cv-corpus-25.0-2026-03-09/yo",
    },
    {
        "key": "hau",
        "config": "hau_tts",
        "language": "hausa",
        "local_dir": "datasets/cmn1qen4q00xjo107gln14ztz/cv-corpus-25.0-2026-03-09/ha",
    },
    {
        "key": "ibo",
        "config": "ibo_tts",
        "language": None,
        "local_dir": "datasets/cmn2cp3yv01h6mm07x6tl0t1i/cv-corpus-25.0-2026-03-09/ig",
    },
    {
        "key": "pcm",
        "config": "pcm_tts",
        "language": "english",
        "local_dir": "datasets/cmn2cgr3101g2mm07mt1zagmz/cv-corpus-25.0-2026-03-09/pcm",
    },
]

RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Text normalisation
# ---------------------------------------------------------------------------
def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text).lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Data collator
# ---------------------------------------------------------------------------
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        input_features = [
            {"input_features": f["input_features"]} for f in features
        ]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(
            label_features, return_tensors="pt"
        )
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


# ---------------------------------------------------------------------------
# WER helper
# ---------------------------------------------------------------------------
def compute_wer_single(reference: str, hypothesis: str, metric) -> float:
    """WER for one sample; clamp to 0 for empty reference."""
    ref = reference.strip()
    if not ref:
        return 0.0
    return metric.compute(predictions=[hypothesis], references=[ref])


# ---------------------------------------------------------------------------
# Build dataset for one language
# ---------------------------------------------------------------------------
def build_dataset(lang_cfg: dict, processor, task: str):
    cfg = lang_cfg["config"]
    lang = lang_cfg["language"]

    ds_hf = load_dataset(DATASET_NAME, cfg, split="test").cast_column(
        "audio", Audio(sampling_rate=SAMPLE_RATE)
    )
    ds_hf = ds_hf.select_columns(["audio", "text"])
    datasets_to_concat = [ds_hf]

    local_dir = lang_cfg.get("local_dir")
    if local_dir and os.path.exists(local_dir):
        local_subsets = []
        for split in ["train", "test"]:
            tsv_path = os.path.join(local_dir, f"{split}.tsv")
            if os.path.exists(tsv_path):
                ds_sub = load_dataset(
                    "csv", data_files=tsv_path, delimiter="\t", split="train"
                )
                ds_sub = ds_sub.select_columns(["path", "sentence"])
                local_subsets.append(ds_sub)

        if local_subsets:
            ds_local = concatenate_datasets(local_subsets)
            splits_1 = ds_local.train_test_split(train_size=0.70, seed=RANDOM_SEED)
            splits_2 = splits_1["test"].train_test_split(train_size=0.50, seed=RANDOM_SEED)
            ds_local_test = splits_2["test"]

            clips_dir = os.path.join(local_dir, "clips")

            def _add_audio_path(batch):
                batch["audio"] = os.path.join(clips_dir, str(batch["path"]))
                batch["text"] = str(batch["sentence"])
                return batch

            ds_local_test = ds_local_test.map(_add_audio_path)
            ds_local_test = ds_local_test.cast_column(
                "audio", Audio(sampling_rate=SAMPLE_RATE)
            )
            ds_local_test = ds_local_test.select_columns(["audio", "text"])
            datasets_to_concat.append(ds_local_test)

    ds = concatenate_datasets(datasets_to_concat)

    if lang is not None:
        processor.tokenizer.set_prefix_tokens(language=lang, task=task)
    else:
        processor.tokenizer.set_prefix_tokens(language=None, task=task, predict_timestamps=False)

    def prepare_dataset(batch):
        audio = batch["audio"]
        batch["input_features"] = processor.feature_extractor(
            audio["array"], sampling_rate=audio["sampling_rate"]
        ).input_features[0]
        batch["labels"] = processor.tokenizer(
            unicodedata.normalize("NFC", batch["text"])
        ).input_ids
        return batch

    ds = ds.map(
        prepare_dataset,
        remove_columns=ds.column_names,
        num_proc=1 if os.name == "nt" else 4,
    )
    return ds


# ---------------------------------------------------------------------------
# Evaluate one language, return result dict
# ---------------------------------------------------------------------------
def evaluate_language(lang_cfg: dict, model, processor, data_collator, metric, device: str):
    lang = lang_cfg["language"]
    lang_str = str(lang).upper() if lang else "NONE"
    cfg = lang_cfg["config"]

    print(f"\n{'='*60}")
    print(f"Evaluating: {lang_str} ({cfg})")
    print(f"{'='*60}")

    ds = build_dataset(lang_cfg, processor, TASK)

    if lang is not None:
        forced_decoder_ids = processor.get_decoder_prompt_ids(language=lang, task=TASK)
    else:
        forced_decoder_ids = processor.get_decoder_prompt_ids(language=None, task=TASK)
        if hasattr(processor.tokenizer, "prefix_tokens"):
            forced_decoder_ids = [
                [i, token_id]
                for i, token_id in enumerate(processor.tokenizer.prefix_tokens)
            ]

    eval_dataloader = DataLoader(ds, batch_size=BATCH_SIZE, collate_fn=data_collator)

    samples = []
    t_start = time.perf_counter()

    for step, batch in enumerate(tqdm(eval_dataloader, desc=f"Eval {lang_str}")):
        with torch.amp.autocast(device):
            with torch.no_grad():
                generated_tokens = (
                    model.generate(
                        input_features=batch["input_features"].to(device),
                        max_new_tokens=255,
                        forced_decoder_ids=forced_decoder_ids,
                    )
                    .cpu()
                    .numpy()
                )

        labels = batch["labels"].cpu().numpy()
        labels = np.where(labels != -100, labels, processor.tokenizer.pad_token_id)

        decoded_preds = processor.tokenizer.batch_decode(
            generated_tokens, skip_special_tokens=True
        )
        decoded_labels = processor.tokenizer.batch_decode(labels, skip_special_tokens=True)

        for i, (ref, hyp) in enumerate(zip(decoded_labels, decoded_preds)):
            sample_id = f"{lang_cfg['key']}_{step * BATCH_SIZE + i:04d}"
            wer_val = compute_wer_single(ref, hyp, metric)
            samples.append(
                {
                    "id": sample_id,
                    "reference": ref,
                    "hypothesis": hyp,
                    "normalized_reference": normalize_text(ref),
                    "normalized_hypothesis": normalize_text(hyp),
                    "wer": wer_val,
                }
            )

        if step == 0:
            print(f"\n[SAMPLE] REF : {decoded_labels[0]}")
            print(f"[SAMPLE] HYP : {decoded_preds[0]}\n")

        del generated_tokens, labels, batch
        gc.collect()

    elapsed = time.perf_counter() - t_start

    # Aggregate metrics
    valid = [s for s in samples if s["reference"].strip()]
    wers = [s["wer"] for s in valid]
    mean_wer = float(np.mean(wers)) if wers else 0.0
    median_wer = float(np.median(wers)) if wers else 0.0

    norm_preds = [s["normalized_hypothesis"] for s in valid if s["normalized_reference"].strip()]
    norm_refs = [s["normalized_reference"] for s in valid if s["normalized_reference"].strip()]
    norm_wer = (
        metric.compute(predictions=norm_preds, references=norm_refs) if norm_refs else 0.0
    )

    result = {
        "language": lang_str,
        "config": cfg,
        "model": DEFAULT_PEFT_MODEL_ID,
        "dataset": DATASET_NAME,
        "n_samples": len(valid),
        "mean_wer": mean_wer,
        "median_wer": median_wer,
        "normalized_wer": float(norm_wer),
        "elapsed_s": elapsed,
        "samples": samples,
    }

    print(f"\n--> {lang_str} WER : {mean_wer*100:.2f}%  |  Normalized WER : {norm_wer*100:.2f}%")
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate Whisper LoRA fine-tune")
    parser.add_argument(
        "--model",
        default=DEFAULT_PEFT_MODEL_ID,
        help="Path to PEFT/LoRA adapter directory",
    )
    parser.add_argument(
        "--langs",
        nargs="+",
        choices=["yor", "hau", "ibo", "pcm"],
        default=None,
        help="Languages to evaluate (default: all four)",
    )
    parser.add_argument(
        "--out-dir",
        default=".",
        help="Directory to write test_results_<lang>.json files",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="Inference batch size",
    )
    args = parser.parse_args()

    global BATCH_SIZE, DEFAULT_PEFT_MODEL_ID
    BATCH_SIZE = args.batch_size
    DEFAULT_PEFT_MODEL_ID = args.model

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Filter languages
    lang_cfgs = LANGUAGE_CONFIGS
    if args.langs:
        lang_cfgs = [c for c in LANGUAGE_CONFIGS if c["key"] in args.langs]

    # ── Load model ──────────────────────────────────────────────────────────
    print(f"Loading PEFT config from {args.model} ...")
    peft_config = PeftConfig.from_pretrained(args.model)
    processor = WhisperProcessor.from_pretrained(
        peft_config.base_model_name_or_path, task=TASK
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    if device == "cuda":
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        base_model = WhisperForConditionalGeneration.from_pretrained(
            peft_config.base_model_name_or_path,
            quantization_config=quantization_config,
            device_map="auto",
        )
    else:
        base_model = WhisperForConditionalGeneration.from_pretrained(
            peft_config.base_model_name_or_path
        ).to(device)

    print("Attaching LoRA adapters ...")
    model = PeftModel.from_pretrained(base_model, args.model)
    model.config.use_cache = True
    model.generation_config.forced_decoder_ids = None
    model.generation_config.suppress_tokens = []
    for attr in ("forced_decoder_ids", "suppress_tokens"):
        if hasattr(model.config, attr):
            delattr(model.config, attr)
    model.eval()

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    metric = evaluate.load("wer")

    # ── Evaluate each language ───────────────────────────────────────────────
    all_preds, all_refs, all_norm_preds, all_norm_refs = [], [], [], []

    for lang_cfg in lang_cfgs:
        result = evaluate_language(lang_cfg, model, processor, data_collator, metric, device)

        out_path = out_dir / f"test_results_{lang_cfg['key']}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"Saved → {out_path}")

        for s in result["samples"]:
            if s["reference"].strip():
                all_preds.append(s["hypothesis"])
                all_refs.append(s["reference"])
            if s["normalized_reference"].strip():
                all_norm_preds.append(s["normalized_hypothesis"])
                all_norm_refs.append(s["normalized_reference"])

    # ── Overall summary ──────────────────────────────────────────────────────
    if all_refs:
        final_wer = 100 * metric.compute(predictions=all_preds, references=all_refs)
        final_norm_wer = 100 * metric.compute(
            predictions=all_norm_preds, references=all_norm_refs
        )
        print("\n" + "=" * 72)
        print(f"Overall WER            : {final_wer:.2f}%")
        print(f"Overall Normalized WER : {final_norm_wer:.2f}%")
        print("=" * 72)


if __name__ == "__main__":
    main()
