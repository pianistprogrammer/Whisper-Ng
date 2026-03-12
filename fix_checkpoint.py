#!/usr/bin/env python3
"""
Fix checkpoint by copying required config files from base model.

When training with HuggingFace Trainer, the checkpoint may be missing
preprocessor_config.json and other config files needed for inference.

This script copies them from the base model.
"""

import sys
import shutil
from pathlib import Path

def fix_checkpoint(checkpoint_path: str, base_model: str = "openai/whisper-base"):
    """
    Copy missing config files from base model to checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint directory
        base_model: Base model to copy configs from
    """
    checkpoint_dir = Path(checkpoint_path)
    
    if not checkpoint_dir.exists():
        print(f"✗ Checkpoint not found: {checkpoint_path}")
        sys.exit(1)
    
    print(f"Fixing checkpoint: {checkpoint_dir}", flush=True)
    
    # Download base model configs to temp directory
    print(f"📥 Downloading base model configs from: {base_model}", flush=True)
    
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    
    try:
        processor = WhisperProcessor.from_pretrained(base_model)
        model = WhisperForConditionalGeneration.from_pretrained(base_model)
    except Exception as e:
        print(f"✗ Failed to load base model: {e}")
        sys.exit(1)
    
    # Save processor (which includes preprocessor_config)
    print(f"\nCopying required files to checkpoint...", flush=True)
    processor.save_pretrained(checkpoint_dir)
    
    # Also save feature extractor separately (for preprocessor_config)
    processor.feature_extractor.save_pretrained(checkpoint_dir)
    
    # Copy preprocessor config from feature extractor
    import json
    
    # Get feature extractor config
    fe_config = processor.feature_extractor.to_dict()
    
    # Save as preprocessor_config.json
    preprocessor_config_path = checkpoint_dir / "preprocessor_config.json"
    with open(preprocessor_config_path, "w") as f:
        json.dump(fe_config, f, indent=2)
    
    print(f"✓ Saved processor, feature_extractor, and preprocessor configs", flush=True)
    
    print(f"\n✓ Checkpoint fixed successfully!", flush=True)
    print(f"  You can now use: python transcribe_with_finetuned.py --checkpoint {checkpoint_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Fix checkpoint configs")
    parser.add_argument("checkpoint", help="Path to checkpoint directory")
    parser.add_argument("--base-model", default="openai/whisper-base",
                        help="Base model to copy configs from")
    
    args = parser.parse_args()
    fix_checkpoint(args.checkpoint, args.base_model)
