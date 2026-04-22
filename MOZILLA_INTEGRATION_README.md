# Mozilla Data Collective Integration for Whisper Training

## Overview

This directory contains tools to integrate Mozilla Data Collective datasets with your Whisper fine-tuning pipeline.

## Files Created

1. **mozilla_dataset_loader.py** - Core loader for Mozilla datasets
2. **download_mozilla_datasets.py** - CLI tool to download datasets
3. **train_multilingual_whisper_with_mozilla.py** - Updated training script

## Important: Dataset Requirements

⚠️ **Whisper requires AUDIO datasets with transcriptions**, not just text!

The current dataset downloaded (`cmn3ht40i00eami07lgydmrgg` - English Hausa Parallel Corpus) is a **text-only translation corpus**. It cannot be used for Whisper training.

### What You Need

For Whisper training, you need datasets that contain:
- Audio files (MP3, WAV, etc.)
- Corresponding text transcriptions
- Metadata (train/validation/test splits)

Common Voice datasets from Mozilla are perfect for this!

## Finding the Right Mozilla Datasets

### Step 1: Browse Mozilla Data Collective

Visit: https://mozilladatacollective.com/datasets

### Step 2: Look for Common Voice Datasets

Search for:
- "Common Voice Hausa"
- "Common Voice Yoruba"
- "Common Voice Igbo"

### Step 3: Get Dataset IDs

Once you find the right datasets:
1. Click on the dataset
2. Accept the terms of use through the web interface
3. Copy the dataset ID from the URL
   - Example: `https://mozilladatacollective.com/datasets/abc123def456`
   - Dataset ID: `abc123def456`

### Step 4: Update download_mozilla_datasets.py

Edit the `DATASETS` dictionary:

```python
DATASETS = {
    "hausa": [
        "your-hausa-common-voice-id",
    ],
    "yoruba": [
        "your-yoruba-common-voice-id",
    ],
    "igbo": [
        "your-igbo-common-voice-id",
    ],
}
```

## Usage

### 1. Set Up Credentials

Your `.env` file is already configured with:
```
client_id=mdc_6f832d2fa3bda3f5900b6fed505b26e7
api_key=5e12653d68967fa2e43309528d2f63fb1003c1a9c814e9f8bbe47aa43595cb3f
```

### 2. Accept Terms of Use

**CRITICAL**: You must accept the dataset terms through the web interface first!
1. Go to https://mozilladatacollective.com/datasets/[dataset-id]
2. Read and accept the terms of use
3. Then you can download via API

### 3. Download Datasets

```bash
python download_mozilla_datasets.py
```

This will:
- Download all specified datasets
- Extract them to `mozilla_cache/`
- Show progress and errors

### 4. Update Training Script

Edit `hf/train_multilingual_whisper_with_mozilla.py`:

```python
MOZILLA_DATASET_IDS = [
    "your-hausa-dataset-id",
    "your-yoruba-dataset-id",
    "your-igbo-dataset-id",
]
```

### 5. Run Training

```bash
python hf/train_multilingual_whisper_with_mozilla.py
```

## Dataset Format

The loader currently supports **Common Voice format**:
```
dataset/
├── clips/           # Audio files (MP3)
├── train.tsv        # Training split
├── dev.tsv          # Validation split
└── test.tsv         # Test split
```

Each TSV file contains:
- `path`: Audio filename
- `sentence`: Text transcription
- `client_id`: Speaker ID
- Other metadata

## Current Status

✅ Downloaded: English-Hausa Parallel Corpus (text-only, not usable for Whisper)
❌ Still needed: Audio datasets for Hausa, Yoruba, and Igbo

## Alternative: Use HuggingFace Datasets

If Mozilla doesn't have good audio datasets for these languages, you can:

1. **Stick with HuggingFace only** (current working solution)
   ```bash
   python hf/train_multilingual_whisper_hf.py
   ```

2. **Look for other audio datasets**:
   - Common Voice on HuggingFace: `mozilla-foundation/common_voice_*`
   - African language datasets: `digitallinguistics/african-speech`
   - Custom recordings

3. **Create your own dataset**:
   - Record audio + transcriptions
   - Format as Common Voice structure
   - Use the mozilla_dataset_loader to convert to HF format

## Next Steps

Please provide the **audio dataset IDs** from Mozilla Data Collective, and I'll help integrate them. Make sure to:

1. ✅ Accept terms of use on the website first
2. ✅ Verify they contain audio files (not just text)
3. ✅ Share the dataset IDs with me

## Questions?

- How to find dataset IDs? Look at the URL when viewing a dataset
- Rate limits? 30 downloads per day per organization
- Format issues? The loader supports Common Voice format; other formats need custom parsers
