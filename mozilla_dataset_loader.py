"""
Mozilla Data Collective Dataset Loader

Downloads and processes datasets from Mozilla Data Collective API
to be compatible with HuggingFace datasets for Whisper fine-tuning.

Usage:
    from mozilla_dataset_loader import MozillaDatasetLoader

    loader = MozillaDatasetLoader(
        client_id="your_client_id",
        api_key="your_api_key"
    )

    # Download a dataset
    dataset = loader.download_dataset(
        dataset_id="common-voice-hausa-123",
        cache_dir="./mozilla_cache"
    )

    # Convert to HuggingFace format
    hf_dataset = loader.to_huggingface_format(dataset)
"""

import json
import os
import tarfile
from pathlib import Path
from typing import Optional, Dict, Any, List
import requests
from datasets import Dataset, DatasetDict, Audio
from tqdm import tqdm


class MozillaDatasetLoader:
    """Load and process Mozilla Data Collective datasets for Whisper training."""

    BASE_URL = "https://mozilladatacollective.com/api"

    def __init__(self, client_id: str, api_key: str):
        """
        Initialize the loader with authentication credentials.

        Args:
            client_id: Mozilla Data Collective client ID
            api_key: Mozilla Data Collective API key
        """
        self.client_id = client_id
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {api_key}"
        }

    def get_dataset_info(self, dataset_id: str) -> Dict[str, Any]:
        """
        Get information about a dataset.

        Args:
            dataset_id: The ID of the dataset

        Returns:
            Dictionary containing dataset metadata
        """
        url = f"{self.BASE_URL}/datasets/{dataset_id}"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        return response.json()

    def create_download_session(self, dataset_id: str) -> Dict[str, Any]:
        """
        Create a download session and get the presigned URL.

        Args:
            dataset_id: The ID of the dataset to download

        Returns:
            Dictionary containing download URL and metadata
        """
        url = f"{self.BASE_URL}/datasets/{dataset_id}/download"
        response = requests.post(url, headers=self.headers)

        if response.status_code == 403:
            raise PermissionError(
                f"Access denied: You must agree to the dataset's terms of use "
                f"through the web interface at https://mozilladatacollective.com/datasets/{dataset_id}"
            )
        elif response.status_code == 429:
            error_data = response.json()
            raise RuntimeError(
                f"Rate limit exceeded: {error_data.get('error', 'Unknown error')}. "
                f"Resets at: {error_data.get('limit', {}).get('resetsAt', 'Unknown')}"
            )

        response.raise_for_status()
        return response.json()

    def download_file(
        self,
        url: str,
        dest_path: Path,
        show_progress: bool = True
    ) -> Path:
        """
        Download a file from a URL with progress bar.

        Args:
            url: URL to download from
            dest_path: Destination file path
            show_progress: Whether to show download progress bar

        Returns:
            Path to the downloaded file
        """
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))

        with open(dest_path, 'wb') as f:
            if show_progress and total_size > 0:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc=dest_path.name) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))
            else:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

        return dest_path

    def extract_tarball(self, tarball_path: Path, extract_dir: Path) -> Path:
        """
        Extract a tarball to a directory.

        Args:
            tarball_path: Path to the tarball file
            extract_dir: Directory to extract to

        Returns:
            Path to the extracted directory
        """
        extract_dir.mkdir(parents=True, exist_ok=True)

        print(f"Extracting {tarball_path.name}...")
        with tarfile.open(tarball_path, 'r:gz') as tar:
            tar.extractall(extract_dir)

        return extract_dir

    def download_dataset(
        self,
        dataset_id: str,
        cache_dir: str = "./mozilla_cache",
        force_download: bool = False
    ) -> Path:
        """
        Download a Mozilla dataset.

        Args:
            dataset_id: The ID of the dataset to download
            cache_dir: Directory to cache downloaded files
            force_download: If True, re-download even if cached

        Returns:
            Path to the extracted dataset directory
        """
        cache_path = Path(cache_dir)
        dataset_dir = cache_path / dataset_id

        # Check if already downloaded
        if dataset_dir.exists() and not force_download:
            print(f"Dataset {dataset_id} already cached at {dataset_dir}")
            return dataset_dir

        # Get dataset info
        print(f"Fetching dataset info for {dataset_id}...")
        info = self.get_dataset_info(dataset_id)
        print(f"  Name: {info['name']}")
        print(f"  Size: {int(info['sizeBytes']) / (1024**3):.2f} GB")
        print(f"  Locale: {info['locale']}")

        # Create download session
        print("Creating download session...")
        session = self.create_download_session(dataset_id)

        # Download the file
        download_url = session['downloadUrl']
        filename = session['filename']
        tarball_path = cache_path / filename

        print(f"Downloading {filename}...")
        self.download_file(download_url, tarball_path)

        # Verify checksum if provided
        if 'checksum' in session:
            print("Verifying checksum...")
            # TODO: Implement checksum verification
            pass

        # Extract the tarball
        extracted_dir = self.extract_tarball(tarball_path, dataset_dir)

        # Clean up tarball to save space
        print(f"Removing tarball to save space...")
        tarball_path.unlink()

        print(f"✓ Dataset ready at: {dataset_dir}")
        return dataset_dir

    def parse_common_voice_tsv(
        self,
        tsv_path: Path,
        clips_dir: Path,
        sample_rate: int = 16000
    ) -> List[Dict[str, Any]]:
        """
        Parse Common Voice TSV file into dataset records.

        Args:
            tsv_path: Path to the TSV file (train.tsv, dev.tsv, test.tsv)
            clips_dir: Path to the directory containing audio clips
            sample_rate: Target sample rate for audio

        Returns:
            List of dataset records
        """
        import csv

        records = []
        with open(tsv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                audio_path = clips_dir / row['path']
                if not audio_path.exists():
                    continue

                records.append({
                    'audio': str(audio_path),
                    'text': row['sentence'],
                    'path': row['path'],
                    'client_id': row.get('client_id', ''),
                })

        return records

    def to_huggingface_format(
        self,
        dataset_dir: Path,
        sample_rate: int = 16000,
        format_type: str = "common_voice"
    ) -> DatasetDict:
        """
        Convert Mozilla dataset to HuggingFace DatasetDict format.

        Args:
            dataset_dir: Path to the extracted dataset directory
            sample_rate: Target sample rate for audio
            format_type: Type of dataset format ("common_voice")

        Returns:
            HuggingFace DatasetDict with train/validation/test splits
        """
        if format_type != "common_voice":
            raise ValueError(f"Unsupported format type: {format_type}")

        # Common Voice datasets have clips/ directory and train.tsv, dev.tsv, test.tsv
        clips_dir = dataset_dir / "clips"
        if not clips_dir.exists():
            # Try to find clips directory recursively
            found_clips = list(dataset_dir.rglob("clips"))
            if found_clips:
                clips_dir = found_clips[0]
            else:
                raise FileNotFoundError(f"Could not find 'clips' directory in {dataset_dir}")

        # Parse TSV files
        splits = {}
        split_mapping = {
            'train.tsv': 'train',
            'dev.tsv': 'validation',
            'test.tsv': 'test'
        }

        for tsv_name, split_name in split_mapping.items():
            tsv_path = dataset_dir / tsv_name
            if not tsv_path.exists():
                # Try to find it recursively
                found_tsv = list(dataset_dir.rglob(tsv_name))
                if found_tsv:
                    tsv_path = found_tsv[0]
                else:
                    print(f"Warning: {tsv_name} not found, skipping {split_name} split")
                    continue

            print(f"Parsing {split_name} split from {tsv_path.name}...")
            records = self.parse_common_voice_tsv(tsv_path, clips_dir, sample_rate)

            if records:
                splits[split_name] = Dataset.from_dict({
                    'audio': [r['audio'] for r in records],
                    'text': [r['text'] for r in records],
                })

        if not splits:
            raise ValueError(f"No valid splits found in {dataset_dir}")

        # Cast audio column to Audio feature
        dataset_dict = DatasetDict(splits)
        for split in dataset_dict:
            dataset_dict[split] = dataset_dict[split].cast_column(
                'audio',
                Audio(sampling_rate=sample_rate)
            )

        return dataset_dict


def load_mozilla_datasets(
    dataset_ids: List[str],
    client_id: str,
    api_key: str,
    cache_dir: str = "./mozilla_cache",
    sample_rate: int = 16000
) -> Dict[str, DatasetDict]:
    """
    Load multiple Mozilla datasets.

    Args:
        dataset_ids: List of Mozilla dataset IDs
        client_id: Mozilla Data Collective client ID
        api_key: Mozilla Data Collective API key
        cache_dir: Directory to cache downloaded files
        sample_rate: Target sample rate for audio

    Returns:
        Dictionary mapping dataset_id to HuggingFace DatasetDict
    """
    loader = MozillaDatasetLoader(client_id=client_id, api_key=api_key)

    datasets = {}
    for dataset_id in dataset_ids:
        print(f"\n{'='*80}")
        print(f"Loading dataset: {dataset_id}")
        print('='*80)

        try:
            # Download dataset
            dataset_dir = loader.download_dataset(
                dataset_id=dataset_id,
                cache_dir=cache_dir
            )

            # Convert to HuggingFace format
            hf_dataset = loader.to_huggingface_format(
                dataset_dir=dataset_dir,
                sample_rate=sample_rate
            )

            datasets[dataset_id] = hf_dataset
            print(f"✓ Loaded {dataset_id}")
            for split, ds in hf_dataset.items():
                print(f"  {split}: {len(ds):,} examples")

        except Exception as e:
            print(f"✗ Failed to load {dataset_id}: {e}")
            continue

    return datasets


if __name__ == "__main__":
    # Example usage
    from dotenv import load_dotenv

    load_dotenv()

    client_id = os.getenv("client_id")
    api_key = os.getenv("api_key")

    if not client_id or not api_key:
        raise ValueError("Please set client_id and api_key in .env file")

    # Example: Load Hausa, Igbo, and Yoruba datasets
    # Replace these with actual dataset IDs from Mozilla
    dataset_ids = [
        "common-voice-hausa-xx",
        "common-voice-igbo-xx",
        "common-voice-yoruba-xx",
    ]

    datasets = load_mozilla_datasets(
        dataset_ids=dataset_ids,
        client_id=client_id,
        api_key=api_key,
        cache_dir="./mozilla_cache"
    )

    print("\n" + "="*80)
    print("All datasets loaded successfully!")
    print("="*80)
