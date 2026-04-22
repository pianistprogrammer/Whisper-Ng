"""
Search Mozilla Data Collective for ASR/TTS datasets for Nigerian languages.

This script helps you find audio datasets suitable for Whisper training.
"""

import os
import sys
from pathlib import Path
from mozilla_dataset_loader import MozillaDatasetLoader
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

# Get credentials from .env
client_id = os.getenv("client_id")
api_key = os.getenv("api_key")

if not client_id or not api_key:
    print("ERROR: Missing credentials!")
    sys.exit(1)

def is_audio_dataset(info):
    """
    Check if a dataset is suitable for ASR/TTS (contains audio).

    Args:
        info: Dataset info dictionary

    Returns:
        bool: True if dataset contains audio
    """
    # Check task type
    task = info.get('task', '').upper()
    if task in ['ASR', 'TTS', 'SPEECH']:
        return True

    # Check format
    format_type = info.get('format', '').upper()
    if format_type in ['MP3', 'WAV', 'FLAC', 'OGG', 'M4A']:
        return True

    # Check name/description for audio indicators
    name = info.get('name', '').lower()
    desc = info.get('longDescription', '').lower()

    audio_keywords = ['audio', 'speech', 'voice', 'asr', 'tts', 'spoken']
    if any(keyword in name or keyword in desc for keyword in audio_keywords):
        return True

    return False

def search_dataset(loader, dataset_id):
    """
    Get info about a dataset and check if it's suitable for ASR/TTS.

    Args:
        loader: MozillaDatasetLoader instance
        dataset_id: Dataset ID to check

    Returns:
        tuple: (info dict, is_audio bool)
    """
    try:
        info = loader.get_dataset_info(dataset_id)
        is_audio = is_audio_dataset(info)
        return info, is_audio
    except Exception as e:
        print(f"  ✗ Error fetching {dataset_id}: {e}")
        return None, False

def main():
    """Search for datasets and filter for ASR/TTS suitable ones."""
    loader = MozillaDatasetLoader(client_id=client_id, api_key=api_key)

    print("=" * 80)
    print("MOZILLA DATA COLLECTIVE - ASR/TTS DATASET FINDER")
    print("=" * 80)
    print()

    # Known dataset IDs to check (add more as you discover them)
    # You can add dataset IDs here as you find them on the website
    datasets_to_check = {
        "hausa": [
            "cmnopto3q00t0mf07v2dtc0ej",  # Hausa-TTS-Dataset
            "cmn1qen4q00xjo107gln14ztz",  # Common Voice 25.0 - Hausa
        ],
        "yoruba": [
            "cmo1nlaah0071mk077mw0qhpv",  # Yoruba dataset 1
            "cmn29vsoh019amm07d95id0mo",  # Yoruba dataset 2
        ],
        "igbo": [
            "cmn2cp3yv01h6mm07x6tl0t1i",  # Igbo dataset
        ],
        "pidgin": [
            "cmnykldcz010knu0737o8bgh9",  # Pidgin English dataset 1
            "cmn2cgr3101g2mm07mt1zagmz",  # Pidgin English dataset 2
        ],
    }

    audio_datasets = []
    text_only_datasets = []

    for lang, dataset_ids in datasets_to_check.items():
        if not dataset_ids:
            continue

        print(f"\n{'=' * 80}")
        print(f"CHECKING {lang.upper()} DATASETS")
        print('=' * 80)

        for dataset_id in dataset_ids:
            print(f"\nDataset ID: {dataset_id}")
            print("-" * 80)

            info, is_audio = search_dataset(loader, dataset_id)

            if info:
                print(f"Name:       {info['name']}")
                print(f"Size:       {int(info['sizeBytes']) / (1024**3):.2f} GB")
                print(f"Locale:     {info.get('locale', 'N/A')}")
                print(f"Task:       {info.get('task', 'N/A')}")
                print(f"Format:     {info.get('format', 'N/A')}")
                print(f"License:    {info.get('license', 'N/A')}")

                if is_audio:
                    print(f"✅ SUITABLE FOR WHISPER TRAINING")
                    audio_datasets.append({
                        'language': lang,
                        'id': dataset_id,
                        'name': info['name'],
                        'size_gb': int(info['sizeBytes']) / (1024**3),
                        'task': info.get('task', 'N/A'),
                        'format': info.get('format', 'N/A'),
                    })
                else:
                    print(f"❌ NOT SUITABLE (text-only or non-audio)")
                    text_only_datasets.append({
                        'language': lang,
                        'id': dataset_id,
                        'name': info['name'],
                    })

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    if audio_datasets:
        print(f"\n✅ Found {len(audio_datasets)} AUDIO dataset(s) suitable for Whisper:")
        print()

        for ds in audio_datasets:
            print(f"  [{ds['language'].upper()}] {ds['id']}")
            print(f"    Name:   {ds['name']}")
            print(f"    Size:   {ds['size_gb']:.2f} GB")
            print(f"    Task:   {ds['task']}")
            print(f"    Format: {ds['format']}")
            print()

        # Generate code snippet
        print("\n" + "=" * 80)
        print("COPY THIS TO download_mozilla_datasets.py:")
        print("=" * 80)
        print()
        print("DATASETS = {")

        # Group by language
        by_lang = {}
        for ds in audio_datasets:
            lang = ds['language']
            if lang not in by_lang:
                by_lang[lang] = []
            by_lang[lang].append(ds)

        for lang, datasets in sorted(by_lang.items()):
            print(f'    "{lang}": [')
            for ds in datasets:
                print(f'        "{ds["id"]}",  # {ds["name"]}')
            print('    ],')

        print("}")
        print()

    else:
        print("\n❌ No audio datasets found")

    if text_only_datasets:
        print(f"\n⚠️  Skipped {len(text_only_datasets)} text-only dataset(s):")
        for ds in text_only_datasets:
            print(f"  [{ds['language'].upper()}] {ds['id']} - {ds['name']}")

    print("\n" + "=" * 80)
    print("NEXT STEPS:")
    print("=" * 80)
    print()
    print("1. Browse Mozilla Data Collective for more datasets:")
    print("   https://mozilladatacollective.com/datasets")
    print()
    print("2. Search for:")
    print("   - 'Common Voice' + language name")
    print("   - 'TTS' + language name")
    print("   - 'ASR' + language name")
    print("   - Language name + 'speech'")
    print()
    print("3. Look for these indicators of audio datasets:")
    print("   - Task: ASR, TTS, Speech")
    print("   - Format: MP3, WAV, FLAC")
    print("   - Keywords: 'audio', 'voice', 'speech', 'spoken'")
    print()
    print("4. Copy dataset IDs from URLs and add to this script")
    print()
    print("5. Run this script again to verify they're audio datasets")
    print()
    print("6. Accept terms on the website, then download:")
    print("   python download_mozilla_datasets.py")

if __name__ == "__main__":
    main()
