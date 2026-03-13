"""
Whisper Nigerian Language Transcription — Streamlit Frontend

Records audio from microphone and transcribes using a fine-tuned Whisper model.

Run:
    streamlit run app.py

If using a local checkpoint, first run:
    python fix_checkpoint.py whisper-base-yoruba/checkpoint-1000
"""

import time
import io
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

DEFAULT_CHECKPOINT = "whisper-base-yoruba/checkpoint-1000"
FALLBACK_MODEL     = "openai/whisper-base"

LANGUAGES = {
    "Auto-detect":     None,
    "Yoruba":          "yo",
    "Hausa":           "ha",
    "Igbo":            "ig",
    "Pidgin English":  "en",
    "English":         "en",
}

# ---------------------------------------------------------------------------
# Model loading (cached so it's only done once per session)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_model(model_path: str):
    """Load Whisper model and processor, cached for the session."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = WhisperProcessor.from_pretrained(model_path)
    model = WhisperForConditionalGeneration.from_pretrained(model_path)
    model.to(device)
    model.eval()
    return processor, model, device


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

        model_source = st.radio(
            "Model source",
            ["Local checkpoint", "HuggingFace Hub", "Base model only"],
            index=0,
        )

        if model_source == "Local checkpoint":
            model_path = st.text_input(
                "Checkpoint path",
                value=DEFAULT_CHECKPOINT,
                help="Path to your local HF trainer checkpoint directory",
            )
        elif model_source == "HuggingFace Hub":
            model_path = st.text_input(
                "Model ID",
                value="openai/whisper-base",
                help="e.g. username/whisper-yoruba",
            )
        else:
            model_path = FALLBACK_MODEL

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
        st.caption(f"Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
        st.caption(f"Model: `{model_path}`")

    # ----- Pre-load model -----
    with st.spinner(f"Loading model `{model_path}`…"):
        try:
            _, _, device = load_model(model_path)
            st.success(f"✅ Model ready  ({device.upper()})", icon="✅")
        except Exception as e:
            st.error(
                f"**Failed to load model:** {e}\n\n"
                "If using a local checkpoint, run:\n"
                "```\npython fix_checkpoint.py whisper-base-yoruba/checkpoint-1000\n```",
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
        st.text_area("", value=text, height=150, label_visibility="collapsed")

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
