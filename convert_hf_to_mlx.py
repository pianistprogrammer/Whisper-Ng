#!/usr/bin/env python3
"""
Convert a HuggingFace fine-tuned Whisper checkpoint to MLX format.

The output directory can be used directly with mlx_whisper.transcribe() and
with train_multilingual_whisper_mlx.py as path_or_hf_repo.

Requirements:
    pip install transformers torch numpy

Usage:
    python convert_hf_to_mlx.py \
        --hf-path  multilingual_whisper_hf/checkpoint-1000 \
        --mlx-path multilingual_whisper_mlx

Or convert all checkpoints at once:
    python convert_hf_to_mlx.py --hf-path multilingual_whisper_hf/checkpoint-1000

The script will write:
    <mlx-path>/
        weights.npz   — all model weights in MLX/OpenAI key format
        config.json   — model dimensions expected by mlx_whisper
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import WhisperForConditionalGeneration, WhisperConfig

# ---------------------------------------------------------------------------
# Key-name mapping:  HuggingFace  →  MLX/OpenAI Whisper
# ---------------------------------------------------------------------------

def _enc_block(m: re.Match) -> str:
    i, rest = m.group(1), m.group(2)
    rest = rest.replace("self_attn.q_proj.",         "attn.query.")
    rest = rest.replace("self_attn.k_proj.",         "attn.key.")
    rest = rest.replace("self_attn.v_proj.",         "attn.value.")
    rest = rest.replace("self_attn.out_proj.",       "attn.out.")
    rest = rest.replace("self_attn_layer_norm.",     "attn_ln.")
    rest = rest.replace("fc1.",                      "mlp1.")
    rest = rest.replace("fc2.",                      "mlp2.")
    rest = rest.replace("final_layer_norm.",         "mlp_ln.")
    return f"encoder.blocks.{i}.{rest}"


def _dec_block(m: re.Match) -> str:
    i, rest = m.group(1), m.group(2)
    rest = rest.replace("self_attn.q_proj.",         "attn.query.")
    rest = rest.replace("self_attn.k_proj.",         "attn.key.")
    rest = rest.replace("self_attn.v_proj.",         "attn.value.")
    rest = rest.replace("self_attn.out_proj.",       "attn.out.")
    rest = rest.replace("self_attn_layer_norm.",     "attn_ln.")
    rest = rest.replace("encoder_attn.q_proj.",      "cross_attn.query.")
    rest = rest.replace("encoder_attn.k_proj.",      "cross_attn.key.")
    rest = rest.replace("encoder_attn.v_proj.",      "cross_attn.value.")
    rest = rest.replace("encoder_attn.out_proj.",    "cross_attn.out.")
    rest = rest.replace("encoder_attn_layer_norm.",  "cross_attn_ln.")
    rest = rest.replace("fc1.",                      "mlp1.")
    rest = rest.replace("fc2.",                      "mlp2.")
    rest = rest.replace("final_layer_norm.",         "mlp_ln.")
    return f"decoder.blocks.{i}.{rest}"


# Keys that exist in HF but are not used by MLX (tied/unused)
_SKIP = {
    "model.encoder.embed_tokens.weight",   # not present in encoder (decoder-only vocab)
    "model.encoder.embed_positions.weight", # encoder uses fixed sinusoidal PE (not stored)
    "proj_out.weight",                     # tied to decoder.token_embedding.weight
}


def remap_key(hf_key: str) -> str | None:
    """Return the MLX key for a HF key, or None if it should be skipped."""
    if hf_key in _SKIP:
        return None

    k = hf_key

    # Strip the "model." wrapper
    if k.startswith("model."):
        k = k[len("model."):]

    # ---------- encoder top-level ----------
    k = k.replace("encoder.layer_norm.weight",       "encoder.ln_post.weight")
    k = k.replace("encoder.layer_norm.bias",         "encoder.ln_post.bias")

    # ---------- encoder blocks ----------
    k = re.sub(r"encoder\.layers\.(\d+)\.(.*)", _enc_block, k)

    # ---------- decoder top-level ----------
    k = k.replace("decoder.embed_positions.weight",  "decoder.positional_embedding")
    k = k.replace("decoder.embed_tokens.weight",     "decoder.token_embedding.weight")
    k = k.replace("decoder.layer_norm.weight",       "decoder.ln.weight")
    k = k.replace("decoder.layer_norm.bias",         "decoder.ln.bias")

    # ---------- decoder blocks ----------
    k = re.sub(r"decoder\.layers\.(\d+)\.(.*)", _dec_block, k)

    return k


# ---------------------------------------------------------------------------
# Config mapping:  HuggingFace WhisperConfig  →  MLX dims dict
# ---------------------------------------------------------------------------

def build_mlx_config(hf_cfg: WhisperConfig) -> dict:
    return {
        "n_mels":        hf_cfg.num_mel_bins,
        "n_audio_ctx":   hf_cfg.max_source_positions,
        "n_audio_state": hf_cfg.d_model,
        "n_audio_head":  hf_cfg.encoder_attention_heads,
        "n_audio_layer": hf_cfg.encoder_layers,
        "n_vocab":       hf_cfg.vocab_size,
        "n_text_ctx":    hf_cfg.max_target_positions,
        "n_text_state":  hf_cfg.d_model,
        "n_text_head":   hf_cfg.decoder_attention_heads,
        "n_text_layer":  hf_cfg.decoder_layers,
    }


# ---------------------------------------------------------------------------
# Alignment-head metadata (fixed, architecture-specific, not fine-tuned)
# ---------------------------------------------------------------------------

# Hardcoded defaults per Whisper architecture (from openai/whisper source).
# These are used for word-level timestamps and don't change with fine-tuning.
_ALIGNMENT_HEADS = {
    "base": np.array([[3,1],[4,2],[4,3],[4,7],[5,1],[5,2],[5,4],[5,6]], dtype=np.int32),
    # extend here for small/medium/large if needed
}

def _patch_alignment_heads(weights: dict) -> None:
    """Add alignment_heads to weights dict if not already present."""
    if "alignment_heads" in weights:
        return
    # Detect architecture from number of encoder layers
    encoder_layer_keys = [k for k in weights if k.startswith("encoder.blocks.")]
    n_layers = max(int(k.split(".")[2]) for k in encoder_layer_keys) + 1 if encoder_layer_keys else 0
    arch = {6: "base"}.get(n_layers)
    if arch and arch in _ALIGNMENT_HEADS:
        weights["alignment_heads"] = _ALIGNMENT_HEADS[arch]
        print(f"  ✓ Patched alignment_heads for whisper-{arch} ({weights['alignment_heads'].shape})",
              flush=True)


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------

def convert(hf_path: str, mlx_path: str, dtype: str = "float16") -> None:
    hf_path  = Path(hf_path)
    mlx_path = Path(mlx_path)
    mlx_path.mkdir(parents=True, exist_ok=True)

    np_dtype = np.float16 if dtype == "float16" else np.float32

    print(f"Loading HF model from: {hf_path}", flush=True)
    model = WhisperForConditionalGeneration.from_pretrained(str(hf_path))
    cfg   = model.config
    state = model.state_dict()
    print(f"  {len(state)} HF weight tensors", flush=True)

    # Remap keys
    mlx_weights: dict[str, np.ndarray] = {}
    skipped:     list[str] = []

    for hf_key, tensor in state.items():
        mlx_key = remap_key(hf_key)
        if mlx_key is None:
            skipped.append(hf_key)
            continue
        arr = tensor.detach().float().numpy().astype(np_dtype)
        # PyTorch Conv1d stores weights as (out, in, kernel);
        # MLX Conv1d expects (out, kernel, in) — swap axes 1 and 2.
        if mlx_key in ("encoder.conv1.weight", "encoder.conv2.weight"):
            arr = arr.transpose(0, 2, 1)
        mlx_weights[mlx_key] = arr

    print(f"  {len(mlx_weights)} MLX weight tensors  ({len(skipped)} skipped: {skipped})",
          flush=True)

    # Carry over alignment_heads from the cached reference MLX model (if available).
    # This is a fixed metadata array (not a trained weight) identifying which decoder
    # attention heads are best for word-level timestamps.  It does not change with
    # fine-tuning and is safe to copy from the base model.
    _patch_alignment_heads(mlx_weights)

    # Save weights
    weights_path = mlx_path / "weights.npz"
    np.savez(str(weights_path), **mlx_weights)
    print(f"✓ Saved weights  → {weights_path}", flush=True)

    # Save config
    mlx_cfg = build_mlx_config(cfg)
    cfg_path = mlx_path / "config.json"
    with open(cfg_path, "w") as f:
        json.dump(mlx_cfg, f, indent=2)
    print(f"✓ Saved config   → {cfg_path}", flush=True)
    print(f"\nConfig: {json.dumps(mlx_cfg, indent=2)}", flush=True)

    # Quick sanity check — load with mlx_whisper
    print("\nRunning load sanity check with mlx_whisper...", flush=True)
    try:
        from mlx_whisper.load_models import load_model as mlx_load
        import mlx.core as mx
        loaded = mlx_load(str(mlx_path), dtype=mx.float16)
        mx.eval(loaded.parameters())
        from mlx.utils import tree_flatten
        loaded_keys = {k for k, _ in tree_flatten(loaded.parameters())}
        converted_keys = set(mlx_weights.keys())
        missing = loaded_keys - converted_keys
        extra   = converted_keys - loaded_keys
        if missing:
            print(f"  ⚠ Keys expected by model but NOT in weights.npz ({len(missing)}):", flush=True)
            for k in sorted(missing):
                print(f"     {k}", flush=True)
        if extra:
            print(f"  ⚠ Keys in weights.npz NOT expected by model ({len(extra)}):", flush=True)
            for k in sorted(extra):
                print(f"     {k}", flush=True)
        if not missing and not extra:
            print("  ✓ All keys match perfectly!", flush=True)
        print("  ✓ Model loads successfully in mlx_whisper", flush=True)
    except Exception as e:
        print(f"  ✗ Sanity check failed: {e}", flush=True)
        print("    (weights.npz is still written — this may be a minor issue)", flush=True)

    print(f"\nConversion complete!  Use with mlx_whisper like:")
    print(f'  mlx_whisper.transcribe(audio, path_or_hf_repo="{mlx_path}")')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Convert a HF fine-tuned Whisper checkpoint to MLX format",
    )
    parser.add_argument(
        "--hf-path",
        default="multilingual_whisper_hf/checkpoint-1000",
        help="Path to HF checkpoint directory  (default: multilingual_whisper_hf/checkpoint-1000)",
    )
    parser.add_argument(
        "--mlx-path",
        default=None,
        help="Output MLX directory  (default: <hf-path>-mlx, e.g. checkpoint-1000-mlx)",
    )
    parser.add_argument(
        "--dtype",
        default="float16",
        choices=["float16", "float32"],
        help="Weight dtype to save  (default: float16)",
    )
    args = parser.parse_args()

    mlx_path = args.mlx_path or (str(args.hf_path).rstrip("/") + "-mlx")
    convert(args.hf_path, mlx_path, args.dtype)


if __name__ == "__main__":
    main()
