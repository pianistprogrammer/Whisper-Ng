#!/usr/bin/env python3
"""
Real-time microphone transcription using fine-tuned Whisper model.

Records audio from your microphone and transcribes it using the trained model.

Requirements:
    pip install pyaudio librosa torch transformers sounddevice numpy

Usage:
    python transcribe_microphone.py [--model_id MODEL_ID] [--duration SECONDS] [--device DEVICE]

Examples:
    # Transcribe a 10-second recording using HF model
    python transcribe_microphone.py --duration 10

    # Use custom model from HuggingFace Hub
    python transcribe_microphone.py --model_id pianist/whisper-yoruba-lora --duration 15

    # Specify audio device
    python transcribe_microphone.py --device 0 --duration 10
"""

import sys
import argparse
import numpy as np
import sounddevice as sd
import librosa
from pathlib import Path

import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration

SAMPLE_RATE = 16000
MODEL_NAME = "openai/whisper-base"  # or use your fine-tuned model


def list_audio_devices():
    """List all available audio devices."""
    print("\nAvailable audio devices:")
    print("-" * 60)
    devices = sd.query_devices()
    if isinstance(devices, dict):
        devices = [devices]
    
    for i, device in enumerate(devices):
        if device.get("max_input_channels", 0) > 0:
            print(f"  Device {i}: {device['name']}")
            print(f"    Max input channels: {device['max_input_channels']}")
    print("-" * 60 + "\n")


def record_audio(duration: int, device: int = None) -> np.ndarray:
    """
    Record audio from microphone.
    
    Args:
        duration: Recording duration in seconds
        device: Audio device index (None = default)
    
    Returns:
        Audio array at 16kHz
    """
    print(f"\n🎤 Recording for {duration} seconds...")
    print("   Speak now...", flush=True)
    
    try:
        audio = sd.rec(
            int(SAMPLE_RATE * duration),
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype=np.float32,
            device=device,
            blocksize=int(SAMPLE_RATE / 10)  # 100ms blocks for responsiveness
        )
        sd.wait()
        print("✓ Recording complete\n", flush=True)
        return audio.squeeze()
    except Exception as e:
        print(f"✗ Recording failed: {e}", file=sys.stderr)
        sys.exit(1)


def transcribe(audio: np.ndarray, model_id: str = MODEL_NAME, language: str = None) -> str:
    """
    Transcribe audio using Whisper model.
    
    Args:
        audio: Audio array at 16kHz
        model_id: HuggingFace model ID
        language: Language code (e.g., 'yo' for Yoruba). None = auto-detect
    
    Returns:
        Transcribed text
    """
    print(f"📥 Loading model: {model_id}", flush=True)
    
    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"   Using device: {device.upper()}", flush=True)
    
    processor = WhisperProcessor.from_pretrained(model_id)
    model = WhisperForConditionalGeneration.from_pretrained(model_id)
    model.to(device)
    model.eval()
    
    print(f"✓ Model loaded\n", flush=True)
    
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
        # Set language token if specified
        if language:
            # Language codes for Whisper: https://github.com/openai/whisper/blob/main/whisper/tokenizer.py
            lang_codes = {
                'yo': 12850,  # Yoruba (approximate)
                'hau': 13197,  # Hausa
                'ig': 13156,  # Igbo
                'pcm': 13255,  # Pidgin
                'en': 13246,  # English
            }
            language_token_id = lang_codes.get(language, None)
            
            predicted_ids = model.generate(
                input_features,
                language=language,
                num_beams=5,
                max_length=224,
            )
        else:
            predicted_ids = model.generate(
                input_features,
                num_beams=5,
                max_length=224,
            )
    
    transcription = processor.batch_decode(
        predicted_ids,
        skip_special_tokens=True
    )[0]
    
    print(f"✓ Transcription generated\n", flush=True)
    
    return transcription


def main():
    parser = argparse.ArgumentParser(
        description="Real-time microphone transcription with Whisper"
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default=MODEL_NAME,
        help=f"HuggingFace model ID (default: {MODEL_NAME})"
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
        help="Audio device index (None = default). Use --list-devices to see options."
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="Language code (yo=Yoruba, hau=Hausa, ig=Igbo, pcm=Pidgin, en=English)"
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List available audio devices and exit"
    )
    
    args = parser.parse_args()
    
    # List devices if requested
    if args.list_devices:
        list_audio_devices()
        return
    
    print("=" * 70)
    print("🎙️  WHISPER MICROPHONE TRANSCRIPTION")
    print("=" * 70)
    
    # Record audio
    audio = record_audio(args.duration, device=args.device)
    
    # Transcribe
    try:
        text = transcribe(audio, model_id=args.model_id, language=args.language)
    except Exception as e:
        print(f"✗ Transcription failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Display result
    print("=" * 70)
    print("TRANSCRIPTION RESULT")
    print("=" * 70)
    print(f"\n{text}\n")
    print("=" * 70)
    
    # Save to file
    output_file = Path("transcription.txt")
    with open(output_file, "w") as f:
        f.write(text)
    print(f"✓ Saved to: {output_file}\n")


if __name__ == "__main__":
    main()
