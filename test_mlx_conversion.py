#!/usr/bin/env python3
"""
Test the HF→MLX converted Whisper model on the yor_tts test split.

Usage:
    python test_mlx_conversion.py
    python test_mlx_conversion.py --max-samples 20   # quick smoke test
    python test_mlx_conversion.py --model-path multilingual_whisper_hf/checkpoint-1000-mlx
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
from datasets import load_dataset

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TARGET_SR = 16_000  # Whisper expects 16 kHz


def resample(array: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Simple rational resampling via scipy."""
    if orig_sr == target_sr:
        return array.astype(np.float32)
    from scipy.signal import resample_poly
    from math import gcd
    g = gcd(orig_sr, target_sr)
    up, down = target_sr // g, orig_sr // g
    resampled = resample_poly(array, up, down)
    return resampled.astype(np.float32)


def normalize_text(t: str) -> str:
    """Lower-case, strip punctuation for fair WER comparison."""
    t = t.lower().strip()
    t = re.sub(r"[^\w\s]", "", t, flags=re.UNICODE)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def wer(reference: str, hypothesis: str) -> float:
    """Compute word-error-rate between two strings."""
    r = reference.split()
    h = hypothesis.split()
    if len(r) == 0:
        return 0.0 if len(h) == 0 else 1.0
    # Dynamic programming edit distance
    d = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1):
        d[i][0] = i
    for j in range(len(h) + 1):
        d[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            if r[i - 1] == h[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = 1 + min(d[i - 1][j], d[i][j - 1], d[i - 1][j - 1])
    return d[len(r)][len(h)] / len(r)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Test MLX-converted Whisper on yor_tts")
    parser.add_argument(
        "--model-path",
        default="multilingual_whisper_hf/checkpoint-1000-mlx",
        help="Path to the MLX weights directory",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit to N samples (default: all 177 test samples)",
    )
    parser.add_argument(
        "--language",
        default="yo",
        help="Language hint for Whisper decoder (default: yo = Yoruba)",
    )
    parser.add_argument(
        "--task",
        default="transcribe",
        choices=["transcribe", "translate"],
    )
    args = parser.parse_args()

    model_path = Path(args.model_path)
    if not model_path.exists():
        sys.exit(f"ERROR: model path does not exist: {model_path}\n"
                 "Run  python convert_hf_to_mlx.py  first.")

    # ---- Load test data ----
    print("Loading yor_tts test split...", flush=True)
    ds = load_dataset("google/WaxalNLP", "yor_tts", split="test", trust_remote_code=True)
    total = len(ds)
    n = min(args.max_samples, total) if args.max_samples else total
    print(f"  Using {n}/{total} samples\n", flush=True)

    # ---- Import mlx_whisper ----
    try:
        import mlx_whisper
    except ImportError:
        sys.exit("mlx_whisper not installed.  Run:  pip install mlx-whisper")

    # ---- Run inference ----
    results = []
    wers = []
    t0 = time.time()

    print(f"{'#':>4}  {'Reference':<45}  {'Hypothesis':<45}  {'WER':>6}")
    print("-" * 108)

    for idx in range(n):
        sample = ds[idx]
        ref_raw = sample["text"]
        audio_arr = np.array(sample["audio"]["array"], dtype=np.float64)
        sr = sample["audio"]["sampling_rate"]

        # Resample to 16 kHz
        audio_16k = resample(audio_arr, sr, TARGET_SR)

        # Transcribe
        out = mlx_whisper.transcribe(
            audio_16k,
            path_or_hf_repo=str(model_path),
            language=args.language,
            task=args.task,
            # Anti-hallucination settings
            condition_on_previous_text=False,
            temperature=(0.0, 0.2, 0.4),
            compression_ratio_threshold=1.8,
            logprob_threshold=-0.8,
            verbose=False,
        )
        hyp_raw = out.get("text", "").strip()

        ref_norm = normalize_text(ref_raw)
        hyp_norm = normalize_text(hyp_raw)
        sample_wer = wer(ref_norm, hyp_norm)
        wers.append(sample_wer)

        results.append({
            "id": sample["id"],
            "reference": ref_raw,
            "hypothesis": hyp_raw,
            "wer": round(sample_wer, 4),
        })

        # Pretty-print row (truncated)
        ref_disp = ref_raw[:43] + ".." if len(ref_raw) > 45 else ref_raw.ljust(45)
        hyp_disp = hyp_raw[:43] + ".." if len(hyp_raw) > 45 else hyp_raw.ljust(45)
        bar = "✓" if sample_wer < 0.3 else ("~" if sample_wer < 0.7 else "✗")
        print(f"{idx+1:>4}  {ref_disp}  {hyp_disp}  {sample_wer:>5.2f} {bar}", flush=True)

    elapsed = time.time() - t0

    # ---- Summary ----
    mean_wer  = float(np.mean(wers))
    median_wer = float(np.median(wers))
    perfect   = sum(1 for w in wers if w == 0.0)
    good      = sum(1 for w in wers if w < 0.3)
    terrible  = sum(1 for w in wers if w >= 1.0)

    print("\n" + "=" * 108)
    print(f"Results on  yor_tts  test split  ({n} samples)")
    print(f"  Model          : {model_path}")
    print(f"  Language hint  : {args.language}  |  Task: {args.task}")
    print(f"  Mean  WER      : {mean_wer:.4f}  ({mean_wer*100:.1f}%)")
    print(f"  Median WER     : {median_wer:.4f}  ({median_wer*100:.1f}%)")
    print(f"  Perfect (WER=0): {perfect}/{n}  ({100*perfect/n:.1f}%)")
    print(f"  Good  (WER<30%): {good}/{n}  ({100*good/n:.1f}%)")
    print(f"  Bad  (WER≥100%): {terrible}/{n}  ({100*terrible/n:.1f}%)")
    print(f"  Elapsed        : {elapsed:.1f}s  ({elapsed/n:.2f}s/sample)")

    # ---- Save results ----
    out_path = model_path / "test_results_yor.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "model": str(model_path),
            "dataset": "google/WaxalNLP / yor_tts / test",
            "n_samples": n,
            "mean_wer": round(mean_wer, 4),
            "median_wer": round(median_wer, 4),
            "elapsed_s": round(elapsed, 1),
            "samples": results,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n✓ Full results saved → {out_path}")


if __name__ == "__main__":
    main()
