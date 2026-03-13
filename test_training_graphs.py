#!/usr/bin/env python3
"""
Quick 8-sample training test (2 per language) for 5 epochs.
Runs the full pipeline and produces all 4 training graphs.

Run:
    python test_training_graphs.py
"""

import sys
import gc
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np

# Patch config before importing training module
import train_multilingual_whisper_mlx as T

# ---- Override config for quick test ----
T.BATCH_SIZE   = 2
T.EPOCHS       = 5
T.OUTPUT_DIR   = Path("test_graph_output")
T.RANDOM_SEED  = 42
np.random.seed(42)

from datasets import load_dataset, concatenate_datasets
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
from mlx_whisper.load_models import load_model

SAMPLES_PER_LANG = 2
EPOCHS           = T.EPOCHS
BATCH_SIZE       = T.BATCH_SIZE
OUTPUT_DIR       = T.OUTPUT_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 70)
    print("8-SAMPLE GRAPH TEST  (2 per language × 4 languages)")
    print("=" * 70)

    # ---- Load 2 samples per language ----
    print("\nLoading 2 samples per language...", flush=True)
    all_train, all_val = [], []

    for cfg in T.LANGUAGE_CONFIGS:
        lang, config = cfg["language"], cfg["config"]
        print(f"  {lang.upper()} ({config})", flush=True)
        ds = load_dataset("google/WaxalNLP", config)
        all_train.append(ds["train"].select(range(SAMPLES_PER_LANG)))
        all_val.append(ds["validation"].select(range(SAMPLES_PER_LANG)))

    combined_train = concatenate_datasets(all_train).shuffle(seed=42)
    combined_val   = concatenate_datasets(all_val).shuffle(seed=42)
    print(f"\n  Combined  train={len(combined_train)}  val={len(combined_val)}", flush=True)

    # ---- Preprocess ----
    print("\nPreprocessing...", flush=True)
    train_ds = combined_train.map(T.preprocess, remove_columns=combined_train.column_names)
    val_ds   = combined_val.map(T.preprocess,   remove_columns=combined_val.column_names)
    print(f"  ✓ train={len(train_ds)}  val={len(val_ds)}", flush=True)
    gc.collect()

    # ---- Load model ----
    print("\nLoading Whisper MLX model...", flush=True)
    model = load_model(T.MLX_MODEL_NAME)
    mx.eval(model.parameters())
    print("  ✓ Model loaded", flush=True)

    # ---- LoRA ----
    model = T.apply_lora_to_model(model, T.LORA_LAYERS, T.LORA_RANK, T.LORA_SCALE, T.LORA_TARGETS)
    trainable = sum(p.size for _, p in tree_flatten(model.trainable_parameters()))
    print(f"  ✓ LoRA applied — {trainable:,} trainable parameters", flush=True)

    # ---- Train ----
    optimizer        = optim.Adam(learning_rate=T.LEARNING_RATE)
    loss_and_grad_fn = nn.value_and_grad(model, T.forward_loss)

    history = {
        "epochs": [],
        "train_loss": [],
        "val_loss": [],
        "step_losses": [],
        "step_numbers": [],
    }
    global_step = 0

    print(f"\n{'='*70}")
    print(f"TRAINING ({EPOCHS} epochs × {len(train_ds)} samples)")
    print(f"{'='*70}\n")

    for epoch in range(EPOCHS):
        print(f"[EPOCH {epoch+1}/{EPOCHS}]", flush=True)
        epoch_losses = []

        for features, labels in T.batch_iter(train_ds, BATCH_SIZE, shuffle=True):
            loss, grads = loss_and_grad_fn(model, features, labels)
            T.apply_lora_updates(model, grads, optimizer)
            mx.eval(model.parameters(), optimizer.state)
            loss_val = float(loss.item())
            epoch_losses.append(loss_val)
            global_step += 1
            history["step_losses"].append(loss_val)
            history["step_numbers"].append(global_step)

        avg_train = np.mean(epoch_losses)
        history["train_loss"].append(avg_train)
        history["epochs"].append(epoch + 1)
        print(f"  Train loss : {avg_train:.4f}", flush=True)

        val_losses = []
        for features, labels in T.batch_iter(val_ds, BATCH_SIZE, shuffle=False):
            val_losses.append(float(T.forward_loss(model, features, labels).item()))
        avg_val = np.mean(val_losses)
        history["val_loss"].append(avg_val)
        print(f"  Val loss   : {avg_val:.4f}", flush=True)

    # ---- Save metrics ----
    with open(OUTPUT_DIR / "metrics.json", "w") as f:
        json.dump(history, f, indent=2)

    # ---- Generate graphs ----
    print("\nGenerating graphs...", flush=True)
    T.OUTPUT_DIR   = OUTPUT_DIR
    T.SMOOTH_WINDOW = 3  # small window since we only have a few steps
    T.save_training_graphs(history, OUTPUT_DIR)

    print("\n" + "="*70)
    print("✅  TEST COMPLETE")
    print(f"   Graphs saved to: {OUTPUT_DIR.absolute()}/graphs/")
    print("   Files produced:")
    for p in sorted((OUTPUT_DIR / "graphs").iterdir()):
        print(f"     {p.name}")
    print("="*70)

    # Open the timestamped graphs folder in Finder
    from datetime import datetime
    import subprocess
    # Find the newest graphs_* dir that was just created
    created = sorted((OUTPUT_DIR).glob("graphs_*"), key=lambda p: p.stat().st_mtime)
    if created:
        subprocess.run(["open", str(created[-1])], check=False)


if __name__ == "__main__":
    try:
        mp.set_start_method("fork", force=True)
    except RuntimeError:
        pass
    main()
