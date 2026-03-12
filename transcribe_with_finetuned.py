#!/usr/bin/env python3
"""
Real-time microphone transcription using YOUR fine-tuned Whisper model.

This script loads your fine-tuned Whisper model and transcribes microphone input.

Usage:
    # If you have a local checkpoint
    python transcribe_with_finetuned.py --checkpoint ./checkpoint-1000

    # If you have it on HuggingFace Hub
    python transcribe_with_finetuned.py --model_id your-username/whisper-yoruba

    # Use the base model to test
    python transcribe_with_finetuned.py --base-only
"""

import sys
import argparse
import numpy as np
import sounddevice as sd
from pathlib import Path

import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration

SAMPLE_RATE = 16000


def list_audio_devices():
    """List available audio devices."""
    print("\nAvailable audio devices:")
    print("-" * 60)
    devices = sd.query_devices()
    if isinstance(devices, dict):
        devices = [devices]
    
    for i, device in enumerate(devices):
        if device.get("max_input_channels", 0) > 0:
            print(f"  Device {i}: {device['name']}")
    print("-" * 60 + "\n")


def record_audio(duration: int, device: int = None) -> np.ndarray:
    """Record audio from microphone."""
    print(f"\n🎤 Recording for {duration} seconds...", flush=True)
    print("   Speak now...", flush=True)
    
    try:
        audio = sd.rec(
            int(SAMPLE_RATE * duration),
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype=np.float32,
            device=device,
        )
        sd.wait()
        print("✓ Recording complete\n", flush=True)
        return audio.squeeze()
    except Exception as e:
        print(f"✗ Recording failed: {e}", file=sys.stderr)
        sys.exit(1)


def transcribe_with_model(
    audio: np.ndarray,
    model_path: str = "openai/whisper-base",
    is_local: bool = False,
    language: str = None
) -> str:
    """
    Transcribe audio using Whisper model.
    
    Args:
        audio: Audio array at 16kHz
        model_path: Path to model (local) or HuggingFace model ID
        is_local: Whether model_path is a local checkpoint
        language: Language code
    
    Returns:
        Transcribed text
    """
    print(f"📥 Loading model...", flush=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"   Device: {device.upper()}", flush=True)
    print(f"   Model: {model_path}", flush=True)
    
    try:
        # Load processor and model
        if is_local:
            # Local checkpoint from HF trainer
            processor = WhisperProcessor.from_pretrained(model_path)
            model = WhisperForConditionalGeneration.from_pretrained(model_path)
        else:
            # HuggingFace Hub
            processor = WhisperProcessor.from_pretrained(model_path)
            model = WhisperForConditionalGeneration.from_pretrained(model_path)
        
        model.to(device)
        model.eval()
        print(f"✓ Model loaded successfully\n", flush=True)
    except Exception as e:
        print(f"✗ Failed to load model: {e}\n", file=sys.stderr)
        sys.exit(1)
    
    # Process audio
    print("🔊 Processing audio...", flush=True)
    
    input_features = processor(
        audio,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt"
    ).input_features
    
    input_features = input_features.to(device)
    print(f"✓ Audio processed (shape: {input_features.shape})\n", flush=True)
    
    # Generate transcription
    print("🧠 Generating transcription...", flush=True)
    
    with torch.no_grad():
        predicted_ids = model.generate(
            input_features,
            num_beams=5,
            max_length=224,
        )
    
    transcription = processor.batch_decode(
        predicted_ids,
        skip_special_tokens=True
    )[0]
    
    print(f"✓ Transcription complete\n", flush=True)
    
    return transcription


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe microphone input with fine-tuned Whisper"
    )
    
    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument(
        "--checkpoint",
        type=str,
        help="Path to local HF trainer checkpoint (e.g., ./checkpoint-1000)"
    )
    model_group.add_argument(
        "--model_id",
        type=str,
        help="HuggingFace model ID (e.g., username/whisper-yoruba)"
    )
    model_group.add_argument(
        "--base-only",
        action="store_true",
        help="Use base Whisper model (not fine-tuned)"
    )
    
    parser.add_argument(
        "--duration",
        type=int,
        default=10,
        help="Recording duration in seconds (default: 10)"
    )
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="Audio device index"
    )
    parser.add_argument(
        "--language",
        type=str,
        help="Language code (yo, hau, ig, pcm, en)"
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List audio devices"
    )
    
    args = parser.parse_args()
    
    if args.list_devices:
        list_audio_devices()
        return
    
    # Determine model path
    if args.checkpoint:
        if not Path(args.checkpoint).exists():
            print(f"✗ Checkpoint not found: {args.checkpoint}", file=sys.stderr)
            sys.exit(1)
        model_path = args.checkpoint
        is_local = True
    elif args.model_id:
        model_path = args.model_id
        is_local = False
    else:
        model_path = "openai/whisper-base"
        is_local = False
    
    print("=" * 70)
    print("🎙️  WHISPER MICROPHONE TRANSCRIPTION")
    print("=" * 70)
    
    # Record
    audio = record_audio(args.duration, device=args.device)
    
    # Transcribe
    try:
        text = transcribe_with_model(
            audio,
            model_path=model_path,
            is_local=is_local,
            language=args.language
        )
    except KeyboardInterrupt:
        print("\n\nInterrupted by user", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"✗ Transcription failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Display result
    print("=" * 70)
    print("📝 TRANSCRIPTION")
    print("=" * 70)
    print(f"\n{text}\n")
    print("=" * 70)
    
    # Save
    output_file = Path("transcription_output.txt")
    with open(output_file, "w") as f:
        f.write(text + "\n")
    print(f"\n✓ Saved to: {output_file}\n")


if __name__ == "__main__":
    main()
