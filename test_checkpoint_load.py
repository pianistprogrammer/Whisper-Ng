#!/usr/bin/env python3
"""Quick test to verify checkpoint loads correctly."""

import sys
from pathlib import Path
from transformers import WhisperProcessor, WhisperForConditionalGeneration

checkpoint = "whisper-base-yoruba/checkpoint-1000"

print(f"Testing checkpoint: {checkpoint}")
print(f"Checkpoint exists: {Path(checkpoint).exists()}", flush=True)

print("\n📥 Loading processor...", flush=True)
try:
    processor = WhisperProcessor.from_pretrained(checkpoint)
    print("✓ Processor loaded", flush=True)
except Exception as e:
    print(f"✗ Processor failed: {e}", flush=True)
    sys.exit(1)

print("📥 Loading model...", flush=True)
try:
    model = WhisperForConditionalGeneration.from_pretrained(checkpoint)
    print("✓ Model loaded", flush=True)
except Exception as e:
    print(f"✗ Model failed: {e}", flush=True)
    sys.exit(1)

print("\n✅ Checkpoint loads successfully!")
print("\nYou can now run:")
print(f"  python transcribe_with_finetuned.py --checkpoint {checkpoint}")
