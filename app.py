"""
Whisper Nigerian Language Transcription — Streamlit Frontend

Records audio from microphone and transcribes using a fine-tuned Whisper model.

Run:
    streamlit run app.py

If using a local checkpoint, first run:
    python fix_checkpoint.py multilingual_whisper_hf/checkpoint-1000
"""

import time
import io
from pathlib import Path
import numpy as np
import streamlit as st
import sounddevice as sd
import scipy.io.wavfile as wavfile

import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SAMPLE_RATE = 16000

DEFAULT_HF_CHECKPOINT  = "multilingual_whisper_hf/checkpoint-1000"
FALLBACK_MODEL         = "openai/whisper-small"
DEFAULT_MLX_MODEL      = "mlx-community/whisper-small-mlx"
DEFAULT_MLX_ADAPTERS   = "multilingual_whisper_lora"

LANGUAGES = {
    "Auto-detect":     None,
    "Yoruba":          "yo",
    "Hausa":           "ha",
    "Igbo":            "ig",
    "Pidgin English":  "en",
    "English":         "en",
}

# ---------------------------------------------------------------------------
# HF Model loading
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_model(model_path: str):
    """Load HF Whisper model and processor, cached for the session."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = WhisperProcessor.from_pretrained(model_path)
    model = WhisperForConditionalGeneration.from_pretrained(model_path)
    model.to(device)
    model.eval()
    return processor, model, device


# ---------------------------------------------------------------------------
# MLX Model loading (LoRA adapters)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def _build_mlx_model(mlx_model_name: str, adapter_dir: str, adapter_file: str = "adapters.npz"):
    """Load MLX Whisper base model and apply saved LoRA adapter weights."""
    import mlx.core as mx
    from mlx_whisper.load_models import load_model as mlx_load
    from train_multilingual_whisper_mlx import (
        apply_lora_to_model, LORA_LAYERS, LORA_RANK, LORA_SCALE, LORA_TARGETS,
    )

    model = mlx_load(mlx_model_name, dtype=mx.float16)
    mx.eval(model.parameters())

    adapter_path = Path(adapter_dir) / adapter_file
    has_adapters = adapter_path.exists()
    if has_adapters:
        model = apply_lora_to_model(model, LORA_LAYERS, LORA_RANK, LORA_SCALE, LORA_TARGETS)
        weights = dict(mx.load(str(adapter_path)))
        model.load_weights(list(weights.items()), strict=False)
        mx.eval(model.parameters())

    return model, has_adapters


def transcribe_mlx(
    audio: np.ndarray,
    mlx_model_name: str,
    adapter_dir: str,
    adapter_file: str,
    language_code: str | None,
) -> str:
    """Run MLX Whisper inference, injecting the LoRA model into the transcribe pipeline."""
    import mlx_whisper
    from mlx_whisper.transcribe import ModelHolder

    model, has_adapters = _build_mlx_model(mlx_model_name, adapter_dir, adapter_file)

    # Inject our model (LoRA or base) so mlx_whisper.transcribe picks it up
    cache_key = f"lora:{adapter_dir}/{adapter_file}" if has_adapters else mlx_model_name
    ModelHolder.model = model
    ModelHolder.model_path = cache_key

    decode_opts: dict = {
        "task": "transcribe",
        "fp16": False,                        # adapters saved as float32
        "temperature": (0.0, 0.2, 0.4),       # try greedy first, fall back to sampling
        "condition_on_previous_text": False,  # key fix: prevents repetition spirals
        "compression_ratio_threshold": 1.8,   # fail fast when output is repetitive
        "logprob_threshold": -0.8,            # fail fast when model is uncertain
        "no_speech_threshold": 0.5,
    }
    if language_code:
        decode_opts["language"] = language_code

    result = mlx_whisper.transcribe(audio, path_or_hf_repo=cache_key, **decode_opts)
    text = result["text"].strip()
    # Post-process: collapse any remaining repeated phrases (safety net)
    import re
    text = re.sub(r'(\b\w[\w\s]{1,30})\s+(?:\1\s*){3,}', r'\1 [...]', text).strip()
    return text


# ---------------------------------------------------------------------------
# Audio recording
# ---------------------------------------------------------------------------

def record_audio(duration: int) -> np.ndarray:
    """Record audio from the default microphone."""
    audio = sd.rec(
        int(SAMPLE_RATE * duration),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype=np.float32,
    )
    # Show a live countdown
    progress = st.progress(0, text="🎤 Recording…")
    for i in range(duration):
        time.sleep(1)
        progress.progress((i + 1) / duration, text=f"🎤 Recording… {i + 1}/{duration}s")
    sd.wait()
    progress.empty()
    return audio.squeeze()


def audio_to_wav_bytes(audio: np.ndarray) -> bytes:
    """Convert float32 numpy audio to WAV bytes for st.audio playback."""
    buf = io.BytesIO()
    pcm = (audio * 32767).astype(np.int16)
    wavfile.write(buf, SAMPLE_RATE, pcm)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------

def transcribe(audio: np.ndarray, model_path: str, language_code: str | None) -> str:
    """Run Whisper inference on audio array."""
    processor, model, device = load_model(model_path)

    input_features = processor(
        audio,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
    ).input_features.to(device)

    with torch.no_grad():
        kwargs = dict(num_beams=5, max_length=224)
        if language_code:
            kwargs["language"] = language_code
        predicted_ids = model.generate(input_features, **kwargs)

    return processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="Whisper — Nigerian Language Transcription",
        page_icon="🎙️",
        layout="centered",
    )

    st.title("🎙️ Whisper Nigerian Language Transcription")
    st.caption("Fine-tuned on Yoruba · Hausa · Igbo · Pidgin English")

    # ----- Sidebar settings -----
    with st.sidebar:
        st.header("⚙️ Settings")

        backend = st.radio(
            "Backend",
            ["HuggingFace", "MLX (Apple Silicon)"],
            index=0,
            help="HuggingFace uses PyTorch. MLX runs natively on Apple Silicon with LoRA adapters.",
        )
        use_mlx = backend == "MLX (Apple Silicon)"

        st.divider()

        if use_mlx:
            st.subheader("MLX Model")
            mlx_model_name = st.text_input(
                "Base model (HF repo)",
                value=DEFAULT_MLX_MODEL,
                help="MLX-converted Whisper repo on HuggingFace Hub",
            )
            adapter_dir = st.text_input(
                "LoRA adapter directory",
                value=DEFAULT_MLX_ADAPTERS,
                help="Folder containing adapters.npz produced by train_multilingual_whisper_mlx.py",
            )
            has_best = (Path(adapter_dir) / "best_adapters.npz").exists()
            has_final = (Path(adapter_dir) / "adapters.npz").exists()
            if has_best or has_final:
                adapter_choices = []
                if has_best:
                    adapter_choices.append("best_adapters.npz  (lowest val loss epoch)")
                if has_final:
                    adapter_choices.append("adapters.npz  (final epoch)")
                adapter_file_label = st.radio(
                    "Adapter weights to use",
                    adapter_choices,
                    index=0,
                )
                adapter_file = "best_adapters.npz" if "best" in adapter_file_label else "adapters.npz"
                st.success(f"✅ {adapter_file} found", icon="🧩")
            else:
                adapter_file = "adapters.npz"
                st.warning("adapters.npz not found — will use base model only", icon="⚠️")
            # placeholders so the rest of the code compiles
            model_path = mlx_model_name
        else:
            st.subheader("HuggingFace Model")
            model_source = st.radio(
                "Model source",
                ["Local checkpoint", "HuggingFace Hub", "Base model only"],
                index=0,
            )
            if model_source == "Local checkpoint":
                model_path = st.text_input(
                    "Checkpoint path",
                    value=DEFAULT_HF_CHECKPOINT,
                    help="Path to your local HF trainer checkpoint directory",
                )
            elif model_source == "HuggingFace Hub":
                model_path = st.text_input(
                    "Model ID",
                    value="openai/whisper-small",
                    help="e.g. username/whisper-yoruba",
                )
            else:
                model_path = FALLBACK_MODEL
            mlx_model_name = DEFAULT_MLX_MODEL
            adapter_dir    = DEFAULT_MLX_ADAPTERS
            adapter_file   = "adapters.npz"

        st.divider()

        language = st.selectbox(
            "Language hint",
            list(LANGUAGES.keys()),
            index=0,
            help="Help Whisper choose the right language. 'Auto-detect' works well too.",
        )

        duration = st.slider(
            "Recording duration (seconds)",
            min_value=3,
            max_value=30,
            value=8,
            step=1,
        )

        st.divider()
        if use_mlx:
            st.caption("Device: MPS (Apple Silicon)")
            st.caption(f"Base: `{mlx_model_name}`")
            st.caption(f"Adapters: `{adapter_dir}`")
        else:
            st.caption(f"Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
            st.caption(f"Model: `{model_path}`")

    # ----- Pre-load model -----
    if use_mlx:
        with st.spinner(f"Loading MLX model `{mlx_model_name}`…"):
            try:
                _, has_adapters = _build_mlx_model(mlx_model_name, adapter_dir, adapter_file)
                label = "fine-tuned (LoRA)" if has_adapters else "base only"
                st.success(f"✅ MLX model ready — {label}", icon="✅")
            except Exception as e:
                st.error(f"**Failed to load MLX model:** {e}")
                st.stop()
    else:
        with st.spinner(f"Loading model `{model_path}`…"):
            try:
                _, _, device = load_model(model_path)
                st.success(f"✅ Model ready  ({device.upper()})", icon="✅")
            except Exception as e:
                st.error(
                    f"**Failed to load model:** {e}\n\n"
                    "If using a local checkpoint, run:\n"
                    "```\npython fix_checkpoint.py multilingual_whisper_hf/checkpoint-1000\n```",
                )
                st.stop()

    st.divider()

    # ----- Record & Transcribe -----
    col1, col2 = st.columns([2, 1])

    with col1:
        record_btn = st.button(
            f"🎤 Record {duration}s and Transcribe",
            type="primary",
            use_container_width=True,
        )

    with col2:
        clear_btn = st.button("🗑️ Clear", use_container_width=True)

    if clear_btn:
        for key in ("audio", "transcription"):
            st.session_state.pop(key, None)

    if record_btn:
        st.info(f"Recording for **{duration} seconds** — speak now!", icon="🎙️")

        try:
            audio = record_audio(duration)
        except Exception as e:
            st.error(f"Microphone error: {e}")
            st.stop()

        st.session_state["audio"] = audio

        with st.spinner("Transcribing…"):
            try:
                if use_mlx:
                    text = transcribe_mlx(audio, mlx_model_name, adapter_dir, adapter_file, LANGUAGES[language])
                else:
                    text = transcribe(audio, model_path, LANGUAGES[language])
                st.session_state["transcription"] = text
            except Exception as e:
                st.error(f"Transcription failed: {e}")
                st.stop()

    # ----- Display results -----
    if "audio" in st.session_state:
        st.subheader("🔊 Your Recording")
        wav_bytes = audio_to_wav_bytes(st.session_state["audio"])
        st.audio(wav_bytes, format="audio/wav")

    if "transcription" in st.session_state:
        st.subheader("📝 Transcription")
        text = st.session_state["transcription"]
        st.text_area("Transcription", value=text, height=150, label_visibility="collapsed")

        col_copy, col_download = st.columns(2)
        with col_download:
            st.download_button(
                "⬇️ Download",
                data=text,
                file_name="transcription.txt",
                mime="text/plain",
                use_container_width=True,
            )

    # ----- History -----
    if "history" not in st.session_state:
        st.session_state["history"] = []

    # Append latest result to history (only when new recording finishes)
    if record_btn and "transcription" in st.session_state:
        st.session_state["history"].append(st.session_state["transcription"])

    if st.session_state["history"]:
        with st.expander(f"📋 History ({len(st.session_state['history'])} items)"):
            for i, entry in enumerate(reversed(st.session_state["history"]), 1):
                st.markdown(f"**{i}.** {entry}")


if __name__ == "__main__":
    main()
