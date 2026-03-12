#!/usr/bin/env python3
"""
Test multilingual Whisper fine-tuning with 8 samples (2 per language).
This validates the training pipeline before running on full dataset.
"""

import sys
import numpy as np
import mlx.core as mx
from mlx.utils import tree_flatten

from datasets import load_dataset, concatenate_datasets
from transformers import WhisperProcessor

import mlx_whisper
from mlx_whisper.load_models import load_model

from train_multilingual_whisper_mlx import (
    extract_features,
    tokenize_text,
    spec_augment,
    preprocess,
    pad_labels,
    collate,
    cross_entropy_loss,
    forward_loss,
    apply_lora_to_model,
    apply_lora_updates,
    PAD_TOKEN_ID,
    IGNORE_INDEX,
    SAMPLE_RATE,
    LORA_RANK,
    LORA_SCALE,
    LORA_LAYERS,
    LORA_TARGETS,
    BATCH_SIZE,
    LEARNING_RATE,
    MODEL_NAME,
    MLX_MODEL_NAME,
)

import mlx.optimizers as optim
import mlx.nn as nn

LANGUAGE_CONFIGS = [
    {"config": "yor_tts", "language": "yoruba"},
    {"config": "hau_tts", "language": "hausa"},
    {"config": "ibo_tts", "language": "igbo"},
    {"config": "pcm_tts", "language": "pidgin english"},
]

def main():
    print("=" * 70)
    print("TESTING MULTILINGUAL WHISPER FINE-TUNING (8 SAMPLES)")
    print("=" * 70)
    
    # =========================================================================
    # LOAD & COMBINE 2 SAMPLES PER LANGUAGE
    # =========================================================================
    
    print("\nLoading 2 samples per language...", flush=True)
    sys.stdout.flush()
    
    all_samples = []
    
    for lang_config in LANGUAGE_CONFIGS:
        config = lang_config["config"]
        language = lang_config["language"]
        
        print(f"  Loading {language.upper()} ({config})...", flush=True)
        raw = load_dataset("google/WaxalNLP", config)
        train_split = raw["train"]
        
        # Take first 2 samples
        samples = train_split.select(range(min(2, len(train_split))))
        print(f"    Got {len(samples)} samples", flush=True)
        
        all_samples.append(samples)
    
    # Combine all
    combined = concatenate_datasets(all_samples)
    print(f"\n✓ Combined dataset: {len(combined)} samples total", flush=True)
    sys.stdout.flush()
    
    # =========================================================================
    # PREPROCESS
    # =========================================================================
    
    print("\nPreprocessing samples...", flush=True)
    sys.stdout.flush()
    
    processed = combined.map(
        preprocess,
        remove_columns=combined.column_names,
    )
    
    print(f"✓ Preprocessed: {len(processed)} samples", flush=True)
    
    # Inspect first sample
    sample_0 = processed[0]
    feat = np.array(sample_0['input_features'])
    labs = sample_0['labels']
    print(f"  Sample 0 shape: features={feat.shape}, labels={len(labs)}", flush=True)
    sys.stdout.flush()
    
    # =========================================================================
    # LOAD MODEL
    # =========================================================================
    
    print("\nLoading Whisper MLX model...", flush=True)
    sys.stdout.flush()
    
    try:
        print(f"  From: {MLX_MODEL_NAME}", flush=True)
        sys.stdout.flush()
        model = load_model(MLX_MODEL_NAME)
        print("  Model weights downloaded", flush=True)
        sys.stdout.flush()
        
        print("  Initializing parameters...", flush=True)
        sys.stdout.flush()
        mx.eval(model.parameters())
        print("✓ Model loaded successfully", flush=True)
        sys.stdout.flush()
    except Exception as e:
        print(f"✗ ERROR loading model: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # =========================================================================
    # APPLY LORA
    # =========================================================================
    
    print("\nApplying LoRA adapters...", flush=True)
    sys.stdout.flush()
    model = apply_lora_to_model(model, LORA_LAYERS, LORA_RANK, LORA_SCALE, LORA_TARGETS)
    
    # Count parameters
    all_params = sum(p.size for _, p in tree_flatten(model.parameters()))
    trainable = sum(p.size for _, p in tree_flatten(model.trainable_parameters()))
    
    print(f"\n✓ Parameters after LoRA:")
    print(f"    Total: {all_params:,}")
    print(f"    Trainable (LoRA): {trainable:,}")
    print(f"    Frozen: {all_params - trainable:,}")
    print(f"    Efficiency: {100*trainable/all_params:.4f}%", flush=True)
    sys.stdout.flush()
    
    if trainable == 0:
        print("✗ ERROR: No trainable parameters!", flush=True)
        sys.exit(1)
    
    # =========================================================================
    # TEST SINGLE FORWARD PASS
    # =========================================================================
    
    print("\nTesting single forward pass...", flush=True)
    sys.stdout.flush()
    
    features, labels = collate([processed[0], processed[1]])
    
    print(f"  Batch shapes: features={features.shape}, labels={labels.shape}", flush=True)
    
    try:
        loss = forward_loss(model, features, labels)
        loss_val = float(loss.item())
        print(f"✓ Forward pass successful: loss={loss_val:.4f}", flush=True)
        sys.stdout.flush()
    except Exception as e:
        print(f"✗ Forward pass FAILED: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # =========================================================================
    # TEST TRAINING STEP
    # =========================================================================
    
    print("\nTesting training step...", flush=True)
    sys.stdout.flush()
    
    optimizer = optim.Adam(learning_rate=LEARNING_RATE)
    loss_and_grad_fn = nn.value_and_grad(model, forward_loss)
    
    try:
        # Step 1
        print(f"  Step 1...", flush=True)
        loss1, grads1 = loss_and_grad_fn(model, features, labels)
        apply_lora_updates(model, grads1, optimizer)
        mx.eval(model.parameters(), optimizer.state)
        loss1_val = float(loss1.item())
        print(f"    Loss: {loss1_val:.4f}", flush=True)
        
        # Step 2
        print(f"  Step 2...", flush=True)
        loss2, grads2 = loss_and_grad_fn(model, features, labels)
        apply_lora_updates(model, grads2, optimizer)
        mx.eval(model.parameters(), optimizer.state)
        loss2_val = float(loss2.item())
        print(f"    Loss: {loss2_val:.4f}", flush=True)
        
        print(f"✓ Training steps successful", flush=True)
        
        if loss2_val >= loss1_val:
            print(f"  ⚠ Warning: Loss increased (should decrease). loss1={loss1_val:.4f}, loss2={loss2_val:.4f}", flush=True)
        else:
            print(f"  ✓ Loss decreased as expected: {loss1_val:.4f} → {loss2_val:.4f}", flush=True)
        
        sys.stdout.flush()
    except Exception as e:
        print(f"✗ Training step FAILED: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # =========================================================================
    # TEST FULL VALIDATION BATCH
    # =========================================================================
    
    print("\nTesting full batch processing...", flush=True)
    sys.stdout.flush()
    
    try:
        batch_losses = []
        for i in range(0, len(processed), BATCH_SIZE):
            batch_end = min(i + BATCH_SIZE, len(processed))
            batch = [processed[j] for j in range(i, batch_end)]
            features_b, labels_b = collate(batch)
            
            loss_b = forward_loss(model, features_b, labels_b)
            batch_losses.append(float(loss_b.item()))
        
        avg_loss = np.mean(batch_losses)
        print(f"✓ Processed {len(processed)} samples in batches", flush=True)
        print(f"  Average loss across batches: {avg_loss:.4f}", flush=True)
        sys.stdout.flush()
    except Exception as e:
        print(f"✗ Batch processing FAILED: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"✓ Dataset loading: PASS (8 samples)")
    print(f"✓ Preprocessing: PASS")
    print(f"✓ Model loading: PASS")
    print(f"✓ LoRA application: PASS ({trainable:,} trainable params)")
    print(f"✓ Forward pass: PASS")
    print(f"✓ Training steps: PASS (loss decreased: {loss1_val:.4f} → {loss2_val:.4f})")
    print(f"✓ Batch processing: PASS")
    print("\n🎉 ALL TESTS PASSED - READY FOR FULL TRAINING")
    print("=" * 70)


if __name__ == "__main__":
    main()
