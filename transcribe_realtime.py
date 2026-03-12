#!/usr/bin/env python3
"""
Real-time continuous microphone transcription with Whisper.

Records chunks of audio and transcribes them continuously as you speak.

Usage:
    python transcribe_realtime.py --chunk_size 10
"""

import sys
import argparse
import numpy as np
import sounddevice as sd
from pathlib import Path
from datetime import datetime

import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration

SAMPLE_RATE = 16000


def list_audio_devices():
    """List available audio devices."""
    print("\nAvailable audio devices:")
    devices = sd.query_devices()
    if isinstance(devices, dict):
        devices = [devices]
    for i, device in enumerate(devices):
        if device.get("max_input_channels", 0) > 0:
            print(f"  Device {i}: {device['name']}")
    print()


def transcribe_chunk(
    audio: np.ndarray,
    processor,
    model,
    device: str
) -> str:
    """Transcribe a single audio chunk."""
    input_features = processor(
        audio,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt"
    ).input_features
    
    input_features = input_features.to(device)
    
    with torch.no_grad():
        predicted_ids = model.generate(
            input_features,
            max_length=224,
        )
    
    text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return text.strip()


def main():
    parser = argparse.ArgumentParser(
        description="Real-time continuous microphone transcription"
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="openai/whisper-base",
        help="Model ID or checkpoint path"
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=10,
        help="Chunk duration in seconds (default: 10)"
    )
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="Audio device index"
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
    )
    
    args = parser.parse_args()
    
    if args.list_devices:
        list_audio_devices()
        return
    
    print("\n" + "=" * 70)
    print("🎙️  REAL-TIME WHISPER TRANSCRIPTION")
    print("=" * 70)
    
    # Load model
    print(f"\n📥 Loading model: {args.model_id}", flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"   Device: {device.upper()}", flush=True)
    
    try:
        processor = WhisperProcessor.from_pretrained(args.model_id)
        model = WhisperForConditionalGeneration.from_pretrained(args.model_id)
        model.to(device)
        model.eval()
        print("✓ Model loaded\n", flush=True)
    except Exception as e:
        print(f"✗ Failed to load model: {e}\n", file=sys.stderr)
        sys.exit(1)
    
    # Prepare output file
    output_file = Path("transcription_continuous.txt")
    
    print(f"🎤 Starting continuous transcription ({args.chunk_size}s chunks)...", flush=True)
    print(f"   Press Ctrl+C to stop\n", flush=True)
    print("-" * 70)
    
    chunk_count = 0
    all_text = []
    
    try:
        while True:
            chunk_count += 1
            timestamp = datetime.now().strftime("%H:%M:%S")
            
            # Record chunk
            print(f"[{timestamp}] Recording chunk {chunk_count}...", flush=True)
            audio = sd.rec(
                int(SAMPLE_RATE * args.chunk_size),
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype=np.float32,
                device=args.device,
            )
            sd.wait()
            
            # Transcribe
            print(f"[{timestamp}] Transcribing...", end=" ", flush=True)
            audio = audio.squeeze()
            
            # Skip if too quiet
            if np.max(np.abs(audio)) < 0.01:
                print("(silent)", flush=True)
                continue
            
            text = transcribe_chunk(audio, processor, model, device)
            
            if text:
                print(f"✓\n[{timestamp}] > {text}\n", flush=True)
                all_text.append(f"[{timestamp}] {text}")
                
                # Save incrementally
                with open(output_file, "a") as f:
                    f.write(f"{text}\n")
            else:
                print("(no speech detected)", flush=True)
    
    except KeyboardInterrupt:
        print("\n\n" + "=" * 70)
        print("🛑 Stopped by user\n")
        print("=" * 70)
        
        if all_text:
            print(f"\n📝 Full transcription ({len(all_text)} chunks):\n")
            for line in all_text:
                print(f"  {line}")
        
        print(f"\n✓ Saved to: {output_file}\n")


if __name__ == "__main__":
    main()
