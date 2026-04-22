"""
Download specific Mozilla Data Collective datasets for Nigerian languages.

This script downloads the datasets you've specified and prepares them
for integration with the Whisper training pipeline.
"""

import os
import sys
from pathlib import Path
from mozilla_dataset_loader import MozillaDatasetLoader, load_mozilla_datasets
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get credentials from .env
client_id = os.getenv("client_id")
api_key = os.getenv("api_key")

if not client_id or not api_key:
    print("ERROR: Missing credentials!")
    print("Please ensure your .env file contains:")
    print("  client_id=your_client_id")
    print("  api_key=your_api_key")
    sys.exit(1)

# Dataset IDs to download - AUDIO DATASETS ONLY (ASR/TTS)
# Total: 7 datasets, ~1.67 GB
# Run search_mozilla_datasets.py to verify new datasets have audio
DATASETS = {
    "hausa": [
        "cmnopto3q00t0mf07v2dtc0ej",  # Hausa-TTS-Dataset (270 MB, TTS, MP3)
        "cmn1qen4q00xjo107gln14ztz",  # Common Voice Scripted Speech 25.0 - Hausa (250 MB, ASR, MP3)
    ],
    "yoruba": [
        "cmo1nlaah0071mk077mw0qhpv",  # Yoruba-TTS-Dataset (310 MB, TTS, MP3)
        "cmn29vsoh019amm07d95id0mo",  # Common Voice Scripted Speech 25.0 - Yoruba (160 MB, ASR, MP3)
    ],
    "igbo": [
        "cmn2cp3yv01h6mm07x6tl0t1i",  # Common Voice Scripted Speech 25.0 - Igbo (360 MB, ASR, MP3)
    ],
    "pidgin": [
        "cmnykldcz010knu0737o8bgh9",  # Naija-TTS-Dataset (320 MB, TTS, MP3)
        "cmn2cgr3101g2mm07mt1zagmz",  # Common Voice Scripted Speech 25.0 - Nigerian Pidgin (280 MB, ASR, MP3)
    ],
}

def main():
    """Download all specified datasets."""
    loader = MozillaDatasetLoader(client_id=client_id, api_key=api_key)

    cache_dir = Path("./mozilla_cache")
    cache_dir.mkdir(exist_ok=True)

    print("=" * 80)
    print("MOZILLA DATA COLLECTIVE DATASET DOWNLOADER")
    print("=" * 80)
    print(f"Cache directory: {cache_dir.resolve()}")
    print()

    all_dataset_ids = []
    for lang, ids in DATASETS.items():
        all_dataset_ids.extend(ids)

    if not all_dataset_ids:
        print("No dataset IDs specified. Please add dataset IDs to the DATASETS dict.")
        return

    print(f"Datasets to download: {len(all_dataset_ids)}")
    print()

    successful = []
    failed = []

    for lang, dataset_ids in DATASETS.items():
        if not dataset_ids:
            continue

        print(f"\n{'=' * 80}")
        print(f"DOWNLOADING {lang.upper()} DATASETS")
        print('=' * 80)

        for dataset_id in dataset_ids:
            try:
                print(f"\nDataset ID: {dataset_id}")
                print("-" * 80)

                # Get dataset info first
                info = loader.get_dataset_info(dataset_id)
                print(f"Name: {info['name']}")
                print(f"Size: {int(info['sizeBytes']) / (1024**3):.2f} GB")
                print(f"Locale: {info.get('locale', 'N/A')}")
                print(f"License: {info.get('license', 'N/A')}")
                print()

                # Download
                dataset_dir = loader.download_dataset(
                    dataset_id=dataset_id,
                    cache_dir=str(cache_dir),
                    force_download=False
                )

                successful.append((lang, dataset_id, info['name']))
                print(f"✓ Successfully downloaded: {dataset_id}")

            except PermissionError as e:
                print(f"✗ Permission denied: {e}")
                print(f"  Please accept the terms at: https://mozilladatacollective.com/datasets/{dataset_id}")
                failed.append((lang, dataset_id, "Permission denied"))

            except Exception as e:
                print(f"✗ Failed to download {dataset_id}: {e}")
                failed.append((lang, dataset_id, str(e)))

    # Summary
    print("\n" + "=" * 80)
    print("DOWNLOAD SUMMARY")
    print("=" * 80)

    if successful:
        print(f"\n✓ Successfully downloaded {len(successful)} dataset(s):")
        for lang, dataset_id, name in successful:
            print(f"  [{lang}] {dataset_id} - {name}")

    if failed:
        print(f"\n✗ Failed to download {len(failed)} dataset(s):")
        for lang, dataset_id, reason in failed:
            print(f"  [{lang}] {dataset_id} - {reason}")

    print(f"\nCache directory: {cache_dir.resolve()}")
    print("\nNext steps:")
    print("1. Update MOZILLA_DATASET_IDS in train_multilingual_whisper_with_mozilla.py")
    print("2. Run training: python hf/train_multilingual_whisper_with_mozilla.py")

if __name__ == "__main__":
    main()
