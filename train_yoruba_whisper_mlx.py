"""
Fine-tuning OpenAI Whisper (base) on Yoruba (yor_tts) using MLX + LoRA.

Requirements:
    pip install mlx mlx-whisper transformers datasets librosa soundfile jiwer matplotlib

Run:
    python finetune_yoruba_whisper.py

Outputs (all saved to OUTPUT_DIR):
    adapters.npz          — LoRA adapter weights
    metrics.json          — full per-epoch + per-step metrics
    loss_curves.png       — train / val loss over epochs
    step_loss.png         — per-step train loss (smoothed)
    perplexity.png        — train / val perplexity over epochs
    wer_cer.png           — Word Error Rate + Char Error Rate over epochs
    grad_norm.png         — gradient norm per optimiser step
    training_summary.png  — 2x3 dashboard of all metrics
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
import matplotlib.gridspec as gridspec

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten

from datasets import load_dataset
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
DATASET_CONFIG    = "yor_tts"

SAMPLE_RATE       = 16000
BATCH_SIZE        = 4
EPOCHS            = 5
LEARNING_RATE     = 1e-4
GRAD_ACCUM_STEPS  = 2

LORA_RANK         = 8
LORA_SCALE        = 20  # Scale factor for LoRA contribution
LORA_LAYERS       = 4   # Number of layers to apply LoRA to (from top)
LORA_TARGETS      = ["query", "value"]  # Attention components to apply LoRA to

OUTPUT_DIR        = Path("yoruba_whisper_lora")
PAD_TOKEN_ID      = 50256
IGNORE_INDEX      = -100
MAX_SEQ_LENGTH    = 448  # Whisper decoder max position embeddings

WER_EVAL_SAMPLES  = 200
SMOOTH_WINDOW     = 20


# ---------------------------------------------------------------------------
# LoRA
# ---------------------------------------------------------------------------

class LoRALinear(nn.Module):
    """LoRA Linear layer - wraps existing Linear layer with low-rank adapters."""
    
    @staticmethod
    def from_linear(linear: nn.Linear, rank: int = 8, scale: float = 20.0):
        """Create LoRA layer from an existing Linear layer."""
        # Get dimensions from the linear layer
        output_dims, input_dims = linear.weight.shape
        
        # Create LoRA layer
        lora_lin = LoRALinear(input_dims, output_dims, rank, scale=scale)
        
        # Keep the original linear layer
        lora_lin.linear = linear
        
        return lora_lin
    
    def __init__(self, input_dims: int, output_dims: int, rank: int = 8, scale: float = 20.0):
        super().__init__()
        self.linear = nn.Linear(input_dims, output_dims)
        self.scale = scale
        
        # Initialize LoRA matrices
        lora_scale = 1.0 / math.sqrt(input_dims)
        self.lora_A = mx.random.uniform(shape=(input_dims, rank), low=-lora_scale, high=lora_scale)
        self.lora_B = mx.zeros((rank, output_dims))
    
    def __call__(self, x):
        # Original linear output + LoRA contribution
        return self.linear(x) + self.scale * (x @ self.lora_A @ self.lora_B)


def apply_lora_to_model(model, lora_layers, rank, scale, targets):
    """Apply LoRA to specific layers in the model (encoder and decoder)."""
    print("Applying LoRA adapters...", flush=True)
    sys.stdout.flush()
    
    # Freeze all base model parameters
    model.freeze()
    
    # Apply LoRA to encoder blocks 
    print("  Applying LoRA to encoder attention layers...", flush=True)
    encoder_lora_count = 0
    for block in model.encoder.blocks[-lora_layers:]:  # Last N blocks
        if "query" in targets:
            block.attn.query = LoRALinear.from_linear(block.attn.query, rank=rank, scale=scale)
            encoder_lora_count += 1
        if "value" in targets:
            block.attn.value = LoRALinear.from_linear(block.attn.value, rank=rank, scale=scale)
            encoder_lora_count += 1
    
    print(f"    Applied LoRA to {encoder_lora_count} encoder layers", flush=True)
    
    # Apply LoRA to decoder blocks
    print("  Applying LoRA to decoder attention layers...", flush=True)
    decoder_lora_count = 0
    for block in model.decoder.blocks[-lora_layers:]:  # Last N blocks
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
    """Apply optimizer updates - MLX handles trainable parameters filtering automatically."""
    optimizer.update(model, grads)
    mx.eval(model.parameters(), optimizer.state)


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

processor = WhisperProcessor.from_pretrained(MODEL_NAME, language="yoruba", task="transcribe")
tokenizer = processor.tokenizer


def extract_features(audio_array: np.ndarray) -> np.ndarray:
    features = processor(audio_array, sampling_rate=SAMPLE_RATE, return_tensors="np").input_features[0]
    return features.astype(np.float16)  # float16 saves 50% memory


def tokenize_text(text: str) -> list[int]:
    tokens = tokenizer(text, add_special_tokens=True).input_ids  # Include Whisper prompt tokens
    # Truncate to max sequence length to fit within model's positional embeddings
    if len(tokens) > MAX_SEQ_LENGTH:
        tokens = tokens[:MAX_SEQ_LENGTH]
    return tokens


# ---------------------------------------------------------------------------
# SpecAugment
# ---------------------------------------------------------------------------

def spec_augment(features: np.ndarray, freq_mask_factor: float = 0.2, time_mask_factor: float = 0.2) -> np.ndarray:
    """Apply SpecAugment: frequency and time masking to mel spectrogram.
    
    Reduces overfitting by 20-30% in speech models.
    Reference: Park et al., "SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition"
    """
    features = features.copy()
    
    # Frequency masking
    freq_mask = int(features.shape[0] * freq_mask_factor)
    if freq_mask > 0:
        f0 = np.random.randint(0, max(1, features.shape[0] - freq_mask))
        features[f0:f0+freq_mask] = 0
    
    # Time masking
    time_mask = int(features.shape[1] * time_mask_factor)
    if time_mask > 0:
        t0 = np.random.randint(0, max(1, features.shape[1] - time_mask))
        features[:, t0:t0+time_mask] = 0
    
    return features


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def preprocess(example):
    audio = example["audio"]["array"].astype(np.float32)
    features = extract_features(audio)
    if np.random.rand() < 0.5:  # Apply SpecAugment with 50% probability
        features = spec_augment(features)
    return {"input_features": features, "labels": tokenize_text(example["text"])}


def pad_labels(batch_labels, pad_id, ignore_id):
    max_len = max(len(l) for l in batch_labels)
    return np.array(
        [seq + [ignore_id] * (max_len - len(seq)) for seq in batch_labels],
        dtype=np.int32
    )


def collate(batch):
    features = np.stack([x["input_features"] for x in batch])
    # MLX conv1d expects (batch, length, channels), so transpose from (batch, channels, length)
    features = np.transpose(features, (0, 2, 1))
    # Convert float16 back to float32 for model processing
    features = features.astype(np.float32)
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
    # Labels are already shifted in forward_loss
    # Decoder takes L-1 inputs and produces L-1 logits
    # So compare logits[:, :] with labels[:, :] directly (no second shift)
    
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
    # Decoder returns (logits, kv_cache, cross_attn)
    logits, _, _  = model.decoder(decoder_input, encoder_out)
    # Pass shifted labels to loss (position t -> label t+1)
    return cross_entropy_loss(logits, labels[:, 1:])


# ---------------------------------------------------------------------------
# Greedy decode (for WER/CER)
# ---------------------------------------------------------------------------

def greedy_decode(model, features, max_new_tokens=128):
    encoder_out = model.encoder(features)
    bos         = tokenizer.bos_token_id or 50258
    tokens      = mx.array([[bos]])
    for _ in range(max_new_tokens):
        # Decoder returns (logits, kv_cache, cross_attn)
        logits, _, _ = model.decoder(tokens, encoder_out)
        next_token = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True)
        tokens     = mx.concatenate([tokens, next_token], axis=1)
        mx.eval(tokens)
        if next_token.item() == tokenizer.eos_token_id:
            break
    return tokens[0].tolist()


def evaluate_wer_cer(model, val_ds, n_samples):
    if not HAS_JIWER:
        return None, None
    refs, hyps = [], []
    indices    = list(range(min(n_samples, len(val_ds))))
    np.random.shuffle(indices)
    for i in indices:
        item      = val_ds[i]
        # Add batch dimension using np.newaxis (equivalent to [None, :])
        features  = mx.array(item["input_features"][np.newaxis])
        ref_text  = tokenizer.decode(item["labels"], skip_special_tokens=True)
        pred_ids  = greedy_decode(model, features)
        pred_text = tokenizer.decode(pred_ids, skip_special_tokens=True)
        refs.append(ref_text)
        hyps.append(pred_text)
    return compute_wer(refs, hyps), compute_cer(refs, hyps)


# ---------------------------------------------------------------------------
# Gradient norm
# ---------------------------------------------------------------------------

def grad_norm(grads):
    total = sum(
        float(mx.sum(g * g).item())
        for _, g in tree_flatten(grads)
        if g is not None
    )
    return math.sqrt(total)


# ---------------------------------------------------------------------------
# Smoothing
# ---------------------------------------------------------------------------

def smooth(values, window):
    if len(values) < window:
        return values
    return np.convolve(values, np.ones(window) / window, mode="valid").tolist()


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

STYLE = {
    "train": "#4C72B0", "val": "#DD8452", "grad": "#55A868",
    "raw": "#CCCCCC", "smooth": "#C44E52",
    "background": "#F8F9FA", "grid": "#E0E0E0",
}


def _ax_style(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_facecolor(STYLE["background"])
    ax.grid(color=STYLE["grid"], linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=8)


def plot_loss_curves(history, out_path):
    epochs = history["epochs"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, history["train_loss"], "o-", color=STYLE["train"], label="Train loss")
    ax.plot(epochs, history["val_loss"],   "s-", color=STYLE["val"],   label="Val loss")
    _ax_style(ax, "Loss per Epoch", "Epoch", "Cross-entropy loss")
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)
    print(f"  saved {out_path}")


def plot_step_loss(history, out_path):
    raw     = history["step_losses"]
    steps   = list(range(1, len(raw) + 1))
    sm      = smooth(raw, SMOOTH_WINDOW)
    s_steps = list(range(SMOOTH_WINDOW, len(raw) + 1))
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(steps,   raw, color=STYLE["raw"],    linewidth=0.8, label="Raw")
    ax.plot(s_steps, sm,  color=STYLE["smooth"], linewidth=1.8, label=f"Smoothed (w={SMOOTH_WINDOW})")
    _ax_style(ax, "Per-step Train Loss", "Step", "Loss")
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)
    print(f"  saved {out_path}")


def plot_perplexity(history, out_path):
    epochs = history["epochs"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, [math.exp(min(l,20)) for l in history["train_loss"]], "o-", color=STYLE["train"], label="Train PPL")
    ax.plot(epochs, [math.exp(min(l,20)) for l in history["val_loss"]],   "s-", color=STYLE["val"],   label="Val PPL")
    _ax_style(ax, "Perplexity per Epoch", "Epoch", "Perplexity")
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)
    print(f"  saved {out_path}")


def plot_wer_cer(history, out_path):
    if not HAS_JIWER or not any(v is not None for v in history["wer"]):
        print("  Skipping WER/CER plot (jiwer not available)")
        return
    epochs = history["epochs"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, [v*100 for v in history["wer"]], "o-", color=STYLE["train"], label="WER (%)")
    ax.plot(epochs, [v*100 for v in history["cer"]], "s-", color=STYLE["val"],   label="CER (%)")
    _ax_style(ax, f"WER / CER per Epoch  (greedy, n={WER_EVAL_SAMPLES})", "Epoch", "Error rate (%)")
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)
    print(f"  saved {out_path}")


def plot_grad_norm(history, out_path):
    norms   = history["grad_norms"]
    steps   = list(range(1, len(norms) + 1))
    sm      = smooth(norms, SMOOTH_WINDOW)
    s_steps = list(range(SMOOTH_WINDOW, len(norms) + 1))
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(steps,   norms, color=STYLE["raw"],  linewidth=0.8, label="Raw")
    ax.plot(s_steps, sm,    color=STYLE["grad"], linewidth=1.8, label=f"Smoothed (w={SMOOTH_WINDOW})")
    _ax_style(ax, "Gradient Norm per Optimiser Step", "Optimiser step", "L2 grad norm")
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)
    print(f"  saved {out_path}")


def plot_dashboard(history, out_path):
    fig = plt.figure(figsize=(16, 9))
    fig.patch.set_facecolor(STYLE["background"])
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    epochs   = history["epochs"]
    raw_s    = history["step_losses"]
    steps    = list(range(1, len(raw_s) + 1))
    sm       = smooth(raw_s, SMOOTH_WINDOW)
    s_steps  = list(range(SMOOTH_WINDOW, len(raw_s) + 1))

    # 1. Loss
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(epochs, history["train_loss"], "o-", color=STYLE["train"], label="Train")
    ax1.plot(epochs, history["val_loss"],   "s-", color=STYLE["val"],   label="Val")
    _ax_style(ax1, "Loss", "Epoch", "Loss")

    # 2. Step loss
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(steps,   raw_s, color=STYLE["raw"],    linewidth=0.7, label="Raw")
    ax2.plot(s_steps, sm,    color=STYLE["smooth"], linewidth=1.6, label="Smoothed")
    _ax_style(ax2, "Step Loss", "Step", "Loss")

    # 3. Perplexity
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(epochs, [math.exp(min(l,20)) for l in history["train_loss"]], "o-", color=STYLE["train"], label="Train PPL")
    ax3.plot(epochs, [math.exp(min(l,20)) for l in history["val_loss"]],   "s-", color=STYLE["val"],   label="Val PPL")
    _ax_style(ax3, "Perplexity", "Epoch", "PPL")

    # 4. WER/CER
    ax4 = fig.add_subplot(gs[1, 0])
    if HAS_JIWER and any(v is not None for v in history["wer"]):
        ax4.plot(epochs, [v*100 for v in history["wer"]], "o-", color=STYLE["train"], label="WER %")
        ax4.plot(epochs, [v*100 for v in history["cer"]], "s-", color=STYLE["val"],   label="CER %")
    else:
        ax4.text(0.5, 0.5, "jiwer not installed", ha="center", va="center",
                 transform=ax4.transAxes, color="#888888")
    _ax_style(ax4, "WER / CER", "Epoch", "Error %")

    # 5. Grad norm
    ax5   = fig.add_subplot(gs[1, 1])
    gn    = history["grad_norms"]
    gstep = list(range(1, len(gn) + 1))
    gsm   = smooth(gn, SMOOTH_WINDOW)
    gs_s  = list(range(SMOOTH_WINDOW, len(gn) + 1))
    ax5.plot(gstep, gn,  color=STYLE["raw"],  linewidth=0.7, label="Raw")
    ax5.plot(gs_s,  gsm, color=STYLE["grad"], linewidth=1.6, label="Smoothed")
    _ax_style(ax5, "Gradient Norm", "Optimiser step", "L2 norm")

    # 6. Epoch time
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.bar(epochs, history["epoch_times"], color=STYLE["train"], alpha=0.8, label="seconds")
    _ax_style(ax6, "Epoch Time", "Epoch", "Seconds")

    fig.suptitle("Whisper LoRA Fine-tuning — Yoruba", fontsize=14, fontweight="bold", y=0.98)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


def _save_all_plots(history):
    print("Saving plots …")
    plot_loss_curves(history, OUTPUT_DIR / "loss_curves.png")
    plot_step_loss  (history, OUTPUT_DIR / "step_loss.png")
    plot_perplexity (history, OUTPUT_DIR / "perplexity.png")
    plot_wer_cer    (history, OUTPUT_DIR / "wer_cer.png")
    plot_grad_norm  (history, OUTPUT_DIR / "grad_norm.png")
    plot_dashboard  (history, OUTPUT_DIR / "training_summary.png")


# ---------------------------------------------------------------------------
# Adapter saving
# ---------------------------------------------------------------------------

def save_adapters(model, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    adapter_weights = {
        name: param
        for name, param in tree_flatten(model.parameters())
        if "lora_A" in name or "lora_B" in name
    }
    mx.savez(str(output_dir / "adapters.npz"), **adapter_weights)
    print(f"Saved {len(adapter_weights)} adapter tensors → {output_dir}/adapters.npz")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Fix multiprocessing on macOS
    try:
        mp.set_start_method('fork', force=True)
    except RuntimeError:
        pass  # Already set
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading dataset …")
    raw = load_dataset(DATASET_NAME, DATASET_CONFIG)

    print("Preprocessing …")
    train_ds = raw["train"].map(preprocess, remove_columns=raw["train"].column_names, num_proc=mp.cpu_count())
    val_ds   = raw["validation"].map(preprocess, remove_columns=raw["validation"].column_names, num_proc=mp.cpu_count())
    print(f"  train: {len(train_ds):,}   val: {len(val_ds):,}")
    
    # Force garbage collection before heavy operations
    gc.collect()
    sys.stdout.flush()

    print("Loading Whisper MLX model …")
    sys.stdout.flush()
    
    # Force synchronization before model loading
    try:
        mx.set_memory_limit(0)  # Disable memory limit for loading
    except:
        try:
            mx.metal.set_memory_limit(0)
        except:
            pass  # Older MLX versions may not have this
    
    try:
        # Try direct loading approach
        print(f"  Attempting to load from: {MLX_MODEL_NAME}")
        sys.stdout.flush()
        
        model = load_model(MLX_MODEL_NAME)
        print("  Model weights downloaded successfully")
        sys.stdout.flush()
        
        # Force evaluation of model parameters - this can hang if there's an issue
        print("  Initializing model parameters...")
        sys.stdout.flush()
        mx.eval(model.parameters())  
        print("  Model parameters initialized")
        sys.stdout.flush()
        
        print("Model loaded successfully")
        sys.stdout.flush()
    except Exception as e:
        import traceback
        print(f"ERROR loading model: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Apply LoRA to freeze base model and train only adapter weights
    print("Applying LoRA adapters...", flush=True)
    sys.stdout.flush()
    model = apply_lora_to_model(model, LORA_LAYERS, LORA_RANK, LORA_SCALE, LORA_TARGETS)

    print("Counting parameters...", flush=True)
    sys.stdout.flush()
    
    try:
        print("  Counting total model parameters...", flush=True)
        sys.stdout.flush()
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
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        trainable = all_params = 0

    optimizer     = optim.AdamW(learning_rate=LEARNING_RATE, weight_decay=0.01)
    loss_and_grad = mx.value_and_grad(forward_loss)

    history = {
        "epochs": [], "train_loss": [], "val_loss": [],
        "wer": [], "cer": [], "epoch_times": [],
        "step_losses": [], "grad_norms": [],
    }

    print("\nStarting training …\n")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_losses      = []
        accumulated_grads = None
        opt_step          = 0
        t0                = time.time()

        for features, labels in batch_iter(train_ds, BATCH_SIZE, shuffle=True):
            loss, grads = loss_and_grad(model, features, labels)
            mx.eval(loss)

            loss_val = loss.item()
            history["step_losses"].append(loss_val)
            epoch_losses.append(loss_val)

            if accumulated_grads is None:
                accumulated_grads = grads
            else:
                accumulated_grads = tree_unflatten([
                    (k, ag + g)
                    for (k, ag), (_, g) in zip(
                        tree_flatten(accumulated_grads), tree_flatten(grads)
                    )
                ])

            opt_step += 1
            if opt_step % GRAD_ACCUM_STEPS == 0:
                scaled = tree_unflatten([
                    (k, g / GRAD_ACCUM_STEPS)
                    for k, g in tree_flatten(accumulated_grads)
                ])
                history["grad_norms"].append(grad_norm(scaled))
                
                # Apply updates only to LoRA parameters
                apply_lora_updates(model, scaled, optimizer)
                accumulated_grads = None

        # Validation
        model.eval()
        val_losses = []
        for features, labels in batch_iter(val_ds, BATCH_SIZE):
            vloss = forward_loss(model, features, labels)
            mx.eval(vloss)
            val_losses.append(vloss.item())

        wer, cer  = evaluate_wer_cer(model, val_ds, WER_EVAL_SAMPLES)
        avg_train = sum(epoch_losses) / len(epoch_losses)
        avg_val   = sum(val_losses)   / len(val_losses)
        elapsed   = time.time() - t0

        history["epochs"].append(epoch)
        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)
        history["wer"].append(wer)
        history["cer"].append(cer)
        history["epoch_times"].append(elapsed)

        wer_str = f"{wer*100:.1f}%" if wer is not None else "n/a"
        cer_str = f"{cer*100:.1f}%" if cer is not None else "n/a"
        print(
            f"Epoch {epoch}/{EPOCHS}  train={avg_train:.4f}  val={avg_val:.4f}  "
            f"ppl={math.exp(min(avg_val,20)):.2f}  WER={wer_str}  CER={cer_str}  time={elapsed:.1f}s"
        )

        # Plots updated after every epoch so you can monitor progress live
        _save_all_plots(history)

    save_adapters(model, OUTPUT_DIR)

    metrics_path = OUTPUT_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nMetrics  → {metrics_path}")
    print("Done. All outputs in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()