#!/usr/bin/env python3
"""
Fine-tuning OpenAI Whisper (base) on multiple Nigerian languages using MLX + LoRA.
Combines 4 language datasets:
  - Yoruba (yor_tts)
  - Hausa (hau_tts)
  - Igbo (ibo_tts)
  - Pidgin English (pcm_tts)

Requirements:
    pip install mlx mlx-whisper transformers datasets librosa soundfile jiwer matplotlib

Run:
    python train_multilingual_whisper_mlx.py

Outputs (all saved to OUTPUT_DIR):
    adapters.npz          — LoRA adapter weights
    metrics.json          — full per-epoch + per-step metrics
    loss_curves.png       — train / val loss over epochs
"""

import sys
import json
import math
import time
import gc
import multiprocessing as mp
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten

from datasets import load_dataset, concatenate_datasets
from transformers import WhisperProcessor

import mlx_whisper
from mlx_whisper.load_models import load_model

try:
    from jiwer import wer as compute_wer, cer as compute_cer
    HAS_JIWER = True
except ImportError:
    HAS_JIWER = False
    print("jiwer not found — WER/CER will be skipped. Install with: pip install jiwer")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_NAME        = "openai/whisper-base"
MLX_MODEL_NAME    = "mlx-community/whisper-base-mlx"
DATASET_NAME      = "google/WaxalNLP"

# Language configs to combine
LANGUAGE_CONFIGS = [
    {"config": "yor_tts", "language": "yoruba"},
    {"config": "hau_tts", "language": "hausa"},
    {"config": "ibo_tts", "language": "igbo"},
    {"config": "pcm_tts", "language": "pidgin english"},
]

SAMPLE_RATE       = 16000
BATCH_SIZE        = 2
EPOCHS            = 10
LEARNING_RATE     = 1e-4
GRAD_ACCUM_STEPS  = 2

LORA_RANK         = 8
LORA_SCALE        = 20
LORA_LAYERS       = 4
LORA_TARGETS      = ["query", "value"]

OUTPUT_DIR        = Path("multilingual_whisper_lora")
PAD_TOKEN_ID      = 50256
IGNORE_INDEX      = -100
MAX_SEQ_LENGTH    = 448

WER_EVAL_SAMPLES  = 200
SMOOTH_WINDOW     = 20

# Random seed for reproducibility
RANDOM_SEED       = 42
np.random.seed(RANDOM_SEED)


# ---------------------------------------------------------------------------
# LoRA
# ---------------------------------------------------------------------------

class LoRALinear(nn.Module):
    """LoRA Linear layer - wraps existing Linear layer with low-rank adapters."""
    
    @staticmethod
    def from_linear(linear: nn.Linear, rank: int = 8, scale: float = 20.0):
        """Create LoRA layer from an existing Linear layer."""
        output_dims, input_dims = linear.weight.shape
        lora_lin = LoRALinear(input_dims, output_dims, rank, scale=scale)
        lora_lin.linear = linear
        return lora_lin
    
    def __init__(self, input_dims: int, output_dims: int, rank: int = 8, scale: float = 20.0):
        super().__init__()
        self.linear = nn.Linear(input_dims, output_dims)
        self.scale = scale
        
        lora_scale = 1.0 / math.sqrt(input_dims)
        self.lora_A = mx.random.uniform(shape=(input_dims, rank), low=-lora_scale, high=lora_scale)
        self.lora_B = mx.zeros((rank, output_dims))
    
    def __call__(self, x):
        return self.linear(x) + self.scale * (x @ self.lora_A @ self.lora_B)


def apply_lora_to_model(model, lora_layers, rank, scale, targets):
    """Apply LoRA to specific layers in the model (encoder and decoder)."""
    print("Applying LoRA adapters...", flush=True)
    sys.stdout.flush()
    
    model.freeze()
    
    print("  Applying LoRA to encoder attention layers...", flush=True)
    encoder_lora_count = 0
    for block in model.encoder.blocks[-lora_layers:]:
        if "query" in targets:
            block.attn.query = LoRALinear.from_linear(block.attn.query, rank=rank, scale=scale)
            encoder_lora_count += 1
        if "value" in targets:
            block.attn.value = LoRALinear.from_linear(block.attn.value, rank=rank, scale=scale)
            encoder_lora_count += 1
    
    print(f"    Applied LoRA to {encoder_lora_count} encoder layers", flush=True)
    
    print("  Applying LoRA to decoder attention layers...", flush=True)
    decoder_lora_count = 0
    for block in model.decoder.blocks[-lora_layers:]:
        if "query" in targets:
            block.cross_attn.query = LoRALinear.from_linear(block.cross_attn.query, rank=rank, scale=scale)
            decoder_lora_count += 1
        if "value" in targets:
            block.cross_attn.value = LoRALinear.from_linear(block.cross_attn.value, rank=rank, scale=scale)
            decoder_lora_count += 1
        if "query" in targets:
            block.attn.query = LoRALinear.from_linear(block.attn.query, rank=rank, scale=scale)
            decoder_lora_count += 1
        if "value" in targets:
            block.attn.value = LoRALinear.from_linear(block.attn.value, rank=rank, scale=scale)
            decoder_lora_count += 1
    
    print(f"    Applied LoRA to {decoder_lora_count} decoder layers", flush=True)
    print("LoRA adapters applied successfully", flush=True)
    sys.stdout.flush()
    
    return model


def apply_lora_updates(model, grads, optimizer):
    """Apply optimizer updates."""
    optimizer.update(model, grads)
    mx.eval(model.parameters(), optimizer.state)


# ---------------------------------------------------------------------------
# Feature extraction & preprocessing
# ---------------------------------------------------------------------------

processor = WhisperProcessor.from_pretrained(MODEL_NAME)
tokenizer = processor.tokenizer


def extract_features(audio_array: np.ndarray) -> np.ndarray:
    features = processor(audio_array, sampling_rate=SAMPLE_RATE, return_tensors="np").input_features[0]
    return features.astype(np.float16)


def tokenize_text(text: str) -> list[int]:
    tokens = tokenizer(text, add_special_tokens=True).input_ids
    if len(tokens) > MAX_SEQ_LENGTH:
        tokens = tokens[:MAX_SEQ_LENGTH]
    return tokens


def spec_augment(features: np.ndarray, freq_mask_factor: float = 0.2, time_mask_factor: float = 0.2) -> np.ndarray:
    """Apply SpecAugment: frequency and time masking."""
    features = features.copy()
    freq_mask = int(features.shape[0] * freq_mask_factor)
    if freq_mask > 0:
        f0 = np.random.randint(0, max(1, features.shape[0] - freq_mask))
        features[f0:f0+freq_mask] = 0
    time_mask = int(features.shape[1] * time_mask_factor)
    if time_mask > 0:
        t0 = np.random.randint(0, max(1, features.shape[1] - time_mask))
        features[:, t0:t0+time_mask] = 0
    return features


def preprocess(example):
    audio = example["audio"]["array"].astype(np.float32)
    features = extract_features(audio)
    if np.random.rand() < 0.5:
        features = spec_augment(features)
    return {"input_features": features, "labels": tokenize_text(example["text"])}


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def pad_labels(batch_labels, pad_id, ignore_id):
    max_len = max(len(l) for l in batch_labels)
    return np.array(
        [seq + [ignore_id] * (max_len - len(seq)) for seq in batch_labels],
        dtype=np.int32
    )


def collate(batch):
    features = np.stack([x["input_features"] for x in batch])
    features = np.transpose(features, (0, 2, 1))
    features = features.astype(np.float32)  # Convert float16 back to float32 for model
    labels   = pad_labels([x["labels"] for x in batch], PAD_TOKEN_ID, IGNORE_INDEX)
    return mx.array(features), mx.array(labels)


def batch_iter(dataset, batch_size, shuffle=False):
    indices = list(range(len(dataset)))
    if shuffle:
        np.random.shuffle(indices)
    for start in range(0, len(indices), batch_size):
        yield collate([dataset[int(i)] for i in indices[start:start + batch_size]])


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def cross_entropy_loss(logits, labels):
    """Compute cross-entropy loss with masking."""
    mask = labels != IGNORE_INDEX
    batch_size, seq_len, vocab_size = logits.shape
    
    flat_logits = logits.reshape(-1, vocab_size)
    flat_labels = labels.reshape(-1)
    flat_mask = mask.reshape(-1)

    safe_labels = mx.where(flat_mask, flat_labels, mx.zeros_like(flat_labels))
    losses = nn.losses.cross_entropy(flat_logits, safe_labels, reduction='none')
    masked_losses = mx.where(flat_mask, losses, mx.zeros_like(losses))
    return mx.sum(masked_losses) / mx.maximum(mx.sum(flat_mask.astype(mx.float32)), mx.array(1.0))


def forward_loss(model, features, labels):
    encoder_out   = model.encoder(features)
    decoder_input = labels[:, :-1]
    logits, _, _  = model.decoder(decoder_input, encoder_out)
    return cross_entropy_loss(logits, labels[:, 1:])


# ---------------------------------------------------------------------------
# Graph generation
# ---------------------------------------------------------------------------

def save_training_graphs(history: dict, output_dir: Path):
    """Generate and save 3 training graphs."""
    graphs_dir = output_dir / "graphs"
    graphs_dir.mkdir(exist_ok=True)

    epochs     = history["epochs"]
    train_loss = history["train_loss"]
    val_loss   = history["val_loss"]
    steps      = history["step_numbers"]
    step_loss  = history["step_losses"]

    # ── 1. Epoch-level train vs val loss ──────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs, train_loss, "o-", color="#2196F3", linewidth=2, label="Train loss")
    ax.plot(epochs, val_loss,   "s--", color="#F44336", linewidth=2, label="Val loss")
    best_epoch = epochs[int(np.argmin(val_loss))]
    best_val   = min(val_loss)
    ax.axvline(best_epoch, color="gray", linestyle=":", alpha=0.7, label=f"Best val (epoch {best_epoch})")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Cross-Entropy Loss", fontsize=12)
    ax.set_title("Train vs Validation Loss", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path1 = graphs_dir / "train_val_loss.png"
    fig.savefig(path1, dpi=150)
    plt.close(fig)
    print(f"  ✓ Saved: {path1}")

    # ── 2. Per-step raw loss curve ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(steps, step_loss, color="#9C27B0", alpha=0.4, linewidth=0.8, label="Step loss")
    # Smoothed overlay
    if len(step_loss) >= SMOOTH_WINDOW:
        kernel = np.ones(SMOOTH_WINDOW) / SMOOTH_WINDOW
        smoothed = np.convolve(step_loss, kernel, mode="valid")
        smooth_x = steps[SMOOTH_WINDOW - 1:]
        ax.plot(smooth_x, smoothed, color="#9C27B0", linewidth=2,
                label=f"Smoothed (window={SMOOTH_WINDOW})")
    ax.set_xlabel("Global Step", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("Per-Step Training Loss", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path2 = graphs_dir / "step_loss.png"
    fig.savefig(path2, dpi=150)
    plt.close(fig)
    print(f"  ✓ Saved: {path2}")

    # ── 3. Val loss improvement bar chart ─────────────────────────────────
    fig, ax = plt.subplots(figsize=(max(6, len(epochs) * 0.8 + 2), 5))
    colors = ["#4CAF50" if v == min(val_loss) else "#90CAF9" for v in val_loss]
    bars = ax.bar(epochs, val_loss, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Validation Loss", fontsize=12)
    ax.set_title("Validation Loss per Epoch  (green = best)", fontsize=14, fontweight="bold")
    ax.set_xticks(epochs)
    for bar, v in zip(bars, val_loss):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    path3 = graphs_dir / "val_loss_bars.png"
    fig.savefig(path3, dpi=150)
    plt.close(fig)
    print(f"  ✓ Saved: {path3}")

    # ── 4. Combined 2×2 dashboard ─────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Multilingual Whisper Fine-Tuning\n(Yoruba · Hausa · Igbo · Pidgin)",
                 fontsize=14, fontweight="bold")

    # top-left: train vs val
    axes[0, 0].plot(epochs, train_loss, "o-", color="#2196F3", label="Train")
    axes[0, 0].plot(epochs, val_loss,   "s--", color="#F44336", label="Val")
    axes[0, 0].set_title("Train vs Val Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # top-right: val bar
    axes[0, 1].bar(epochs, val_loss, color=colors, edgecolor="white")
    axes[0, 1].set_title("Val Loss per Epoch")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Val Loss")
    axes[0, 1].set_xticks(epochs)
    axes[0, 1].grid(True, axis="y", alpha=0.3)

    # bottom-left: step loss
    axes[1, 0].plot(steps, step_loss, color="#9C27B0", alpha=0.35, linewidth=0.7)
    if len(step_loss) >= SMOOTH_WINDOW:
        axes[1, 0].plot(smooth_x, smoothed, color="#9C27B0", linewidth=1.8)
    axes[1, 0].set_title("Step Loss (raw + smoothed)")
    axes[1, 0].set_xlabel("Step")
    axes[1, 0].set_ylabel("Loss")
    axes[1, 0].grid(True, alpha=0.3)

    # bottom-right: gap (train - val)
    gap = [t - v for t, v in zip(train_loss, val_loss)]
    gap_colors = ["#EF9A9A" if g < 0 else "#A5D6A7" for g in gap]
    axes[1, 1].bar(epochs, gap, color=gap_colors, edgecolor="white")
    axes[1, 1].axhline(0, color="black", linewidth=0.8)
    axes[1, 1].set_title("Train − Val Gap  (+ = overfitting, − = underfitting)")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Gap")
    axes[1, 1].set_xticks(epochs)
    axes[1, 1].grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    path4 = graphs_dir / "dashboard.png"
    fig.savefig(path4, dpi=150)
    plt.close(fig)
    print(f"  ✓ Saved: {path4}")

    print(f"\n  All graphs saved to: {graphs_dir.absolute()}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    try:
        mp.set_start_method('fork', force=True)
    except RuntimeError:
        pass
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("MULTILINGUAL WHISPER FINE-TUNING (4 NIGERIAN LANGUAGES)")
    print("=" * 70)
    
    # Load all language datasets
    print("\nLoading datasets...", flush=True)
    all_train = []
    all_val = []
    total_train = 0
    total_val = 0
    
    for lang_config in LANGUAGE_CONFIGS:
        config = lang_config["config"]
        language = lang_config["language"]
        
        print(f"  Loading {language.upper()} ({config})...", flush=True)
        raw = load_dataset("google/WaxalNLP", config)
        
        train_split = raw["train"]
        val_split = raw["validation"]
        
        print(f"    Train samples: {len(train_split):,}", flush=True)
        print(f"    Val samples: {len(val_split):,}", flush=True)
        
        all_train.append(train_split)
        all_val.append(val_split)
        total_train += len(train_split)
        total_val += len(val_split)
    
    # Combine datasets
    print(f"\nCombining datasets...", flush=True)
    combined_train = concatenate_datasets(all_train)
    combined_val = concatenate_datasets(all_val)
    
    print(f"  Total train: {len(combined_train):,}", flush=True)
    print(f"  Total val: {len(combined_val):,}", flush=True)
    
    # Shuffle
    print(f"\nShuffling combined dataset...", flush=True)
    combined_train = combined_train.shuffle(seed=RANDOM_SEED)
    combined_val = combined_val.shuffle(seed=RANDOM_SEED)
    print(f"  ✓ Dataset shuffled", flush=True)

    # Preprocess
    print("\nPreprocessing...", flush=True)
    train_ds = combined_train.map(
        preprocess,
        remove_columns=combined_train.column_names,
        num_proc=mp.cpu_count()
    )
    val_ds = combined_val.map(
        preprocess,
        remove_columns=combined_val.column_names,
        num_proc=mp.cpu_count()
    )
    print(f"  Preprocessed train: {len(train_ds):,}", flush=True)
    print(f"  Preprocessed val: {len(val_ds):,}", flush=True)
    
    gc.collect()
    sys.stdout.flush()

    print("\nLoading Whisper MLX model…", flush=True)
    sys.stdout.flush()
    
    try:
        print(f"  Attempting to load from: {MLX_MODEL_NAME}", flush=True)
        sys.stdout.flush()
        
        model = load_model(MLX_MODEL_NAME)
        print("  Model weights downloaded successfully", flush=True)
        sys.stdout.flush()
        
        print("  Initializing model parameters...", flush=True)
        sys.stdout.flush()
        mx.eval(model.parameters())  
        print("  Model parameters initialized", flush=True)
        sys.stdout.flush()
        
        print("Model loaded successfully", flush=True)
        sys.stdout.flush()
    except Exception as e:
        print(f"ERROR loading model: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\nApplying LoRA adapters...", flush=True)
    sys.stdout.flush()
    model = apply_lora_to_model(model, LORA_LAYERS, LORA_RANK, LORA_SCALE, LORA_TARGETS)

    print("\nCounting parameters...", flush=True)
    sys.stdout.flush()
    
    try:
        all_params = sum(p.size for _, p in tree_flatten(model.parameters()))
        trainable = sum(p.size for _, p in tree_flatten(model.trainable_parameters()))
        
        print(f"✓ Parameters:", flush=True)
        print(f"    Total: {all_params:,}", flush=True)
        print(f"    Trainable (LoRA): {trainable:,}", flush=True)
        print(f"    Frozen (base model): {all_params - trainable:,}", flush=True)
        print(f"    Training efficiency: {100*trainable/all_params:.4f}%", flush=True)
        sys.stdout.flush()
    except Exception as e:
        print(f"  Warning: Could not count parameters: {e}", flush=True)

    print("\nTraining configuration:")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Grad accumulation: {GRAD_ACCUM_STEPS}")
    print(f"  Languages: {', '.join([cfg['language'].title() for cfg in LANGUAGE_CONFIGS])}")
    print(f"  Total training samples: {len(train_ds):,}")
    print(f"  Total validation samples: {len(val_ds):,}")
    
    optimizer = optim.Adam(learning_rate=LEARNING_RATE)
    loss_and_grad_fn = nn.value_and_grad(model, forward_loss)
    
    print("\n" + "=" * 70)
    print("STARTING TRAINING")
    print("=" * 70 + "\n")
    
    history = {
        "epochs": [],
        "train_loss": [],
        "val_loss": [],
        "step_losses": [],   # (global_step, loss) for every batch
        "step_numbers": [],
    }
    global_step = 0

    for epoch in range(EPOCHS):
        print(f"\n[EPOCH {epoch+1}/{EPOCHS}]")
        epoch_losses = []
        
        # Training
        print(f"  Training...", flush=True)
        for batch_idx, (features, labels) in enumerate(batch_iter(train_ds, BATCH_SIZE, shuffle=True)):
            loss, grads = loss_and_grad_fn(model, features, labels)
            apply_lora_updates(model, grads, optimizer)
            mx.eval(model.parameters(), optimizer.state)

            loss_val = float(loss.item())
            epoch_losses.append(loss_val)
            global_step += 1
            history["step_losses"].append(loss_val)
            history["step_numbers"].append(global_step)

            if (batch_idx + 1) % 10 == 0:
                avg_loss = np.mean(epoch_losses[-10:])
                print(f"    Step {batch_idx+1:,}: loss={loss_val:.4f} (avg={avg_loss:.4f})", flush=True)
        
        avg_train_loss = np.mean(epoch_losses)
        history["train_loss"].append(avg_train_loss)
        history["epochs"].append(epoch + 1)
        
        print(f"  ✓ Epoch loss: {avg_train_loss:.6f}", flush=True)
        
        # Validation
        print(f"  Validating...", flush=True)
        val_losses = []
        for features, labels in batch_iter(val_ds, BATCH_SIZE, shuffle=False):
            loss = forward_loss(model, features, labels)
            val_losses.append(float(loss.item()))
        
        avg_val_loss = np.mean(val_losses)
        history["val_loss"].append(avg_val_loss)
        
        print(f"  ✓ Val loss: {avg_val_loss:.6f}", flush=True)
        sys.stdout.flush()

    # Save results
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE - SAVING RESULTS")
    print("=" * 70)
    
    # Save LoRA weights
    lora_weights = {}
    for name, p in tree_flatten(model.parameters()):
        if "lora_A" in name or "lora_B" in name:
            lora_weights[name] = p
    
    adapter_path = OUTPUT_DIR / "adapters.npz"
    np.savez(adapter_path, **{k: np.array(v) for k, v in lora_weights.items()})
    print(f"\n✓ Saved adapters to: {adapter_path}")
    
    # Save metrics
    metrics_path = OUTPUT_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"✓ Saved metrics to: {metrics_path}")

    # ------------------------------------------------------------------
    # Generate graphs
    # ------------------------------------------------------------------
    print("\nGenerating training graphs...", flush=True)
    save_training_graphs(history, OUTPUT_DIR)

    print("\n✓ Training complete!")
    print(f"  Output directory: {OUTPUT_DIR.absolute()}")
    print(f"  Graphs saved to: {OUTPUT_DIR}/graphs/")


if __name__ == "__main__":
    main()
