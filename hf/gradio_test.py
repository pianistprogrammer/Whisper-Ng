#!/usr/bin/env python3
"""
Gradio test interface for the fine-tuned HuggingFace Whisper (small) model.

Models available:
  - whisper-small-nigerian/  (whisper-small fine-tuned on all 4 Nigerian languages)
  ... plus every checkpoint-NNNN inside it.

The best checkpoint is highlighted with ★ in the dropdown and is pre-selected.
The [final] model always contains the best checkpoint weights
(saved by load_best_model_at_end=True in the trainer).

Run from the repo root:
    python hf/gradio_test.py

Or from the hf/ folder:
    python gradio_test.py
"""

from pathlib import Path
import numpy as np
import torch
import gradio as gr
from functools import lru_cache
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# ---------------------------------------------------------------------------
# Paths — resolve relative to this file so the script works from anywhere
# ---------------------------------------------------------------------------

HF_DIR = Path(__file__).parent          # .../Whisper-Ng/hf/

MODEL_ROOTS = {
    "whisper-small-nigerian": HF_DIR / "whisper-small-nigerian",
}


def _load_best_info(root: Path) -> dict:
    """Return the contents of best_model_info.json if present, else {}."""
    info_file = root / "best_model_info.json"
    if info_file.exists():
        import json
        with open(info_file) as f:
            return json.load(f)
    return {}

SAMPLE_RATE = 16000

LANGUAGES = {
    "Auto-detect":    None,
    "Yoruba":         "yo",
    "Hausa":          "ha",
    "Igbo":           "ig",
    "Pidgin English": "en",
    "English":        "en",
}

# ---------------------------------------------------------------------------
# Build model list: final + every checkpoint, newest checkpoints first
# ---------------------------------------------------------------------------

def _collect_model_choices() -> list[str]:
    choices = []
    best_default = None
    for label, root in MODEL_ROOTS.items():
        if not root.exists():
            continue
        best_info = _load_best_info(root)
        best_step = best_info.get("best_step")   # int or None

        # Add the final saved model first — this IS the best model
        if (root / "model.safetensors").exists():
            tag = f"{label}  [final ★ best]" if best_step else f"{label}  [final]"
            choices.append(tag)
            if best_default is None:
                best_default = tag   # pre-select this by default

        # Add checkpoints newest → oldest, marking the best one
        ckpts = sorted(
            [d for d in root.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
            key=lambda d: int(d.name.split("-")[1]),
            reverse=True,
        )
        for ckpt in ckpts:
            step = int(ckpt.name.split("-")[1])
            star = " ★ best" if step == best_step else ""
            choices.append(f"{label}  [checkpoint-{step}{star}]")
    return choices if choices else ["No models found — check hf/ folder"], best_default


MODEL_CHOICES, BEST_MODEL_DEFAULT = _collect_model_choices()


def _resolve_path(choice: str) -> Path:
    """Turn a display choice string back into an absolute Path."""
    label, tag = choice.split("  ", 1)          # e.g. "whisper-small-nigerian", "[final ★ best]"
    root = MODEL_ROOTS[label.strip()]
    tag = tag.strip().strip("[]")               # e.g. "final ★ best" or "checkpoint-3000 ★ best"
    # Strip ★ annotation before resolving the path
    tag = tag.replace(" ★ best", "").strip()
    if tag == "final":
        return root
    return root / tag


# ---------------------------------------------------------------------------
# Model loading — cached by path string so we don't reload on every click
# ---------------------------------------------------------------------------

@lru_cache(maxsize=4)
def _load(model_path_str: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = WhisperProcessor.from_pretrained(model_path_str)
    model = WhisperForConditionalGeneration.from_pretrained(model_path_str)
    model.to(device).eval()
    return processor, model, device


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------

def transcribe(audio, model_choice: str, language_label: str) -> tuple[str, str]:
    """
    Parameters
    ----------
    audio : tuple (sample_rate, np.ndarray) from gr.Audio, or None
    model_choice : display string from the dropdown
    language_label : key in LANGUAGES dict

    Returns
    -------
    (transcription_text, status_text)
    """
    if audio is None:
        return "", "⚠️  No audio provided."

    if "No models found" in model_choice:
        return "", "⚠️  No models detected in the hf/ folder."

    sr, waveform = audio

    # Convert to float32 mono
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    waveform = waveform.astype(np.float32)

    # Resample if needed
    if sr != SAMPLE_RATE:
        import librosa
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=SAMPLE_RATE)

    # Normalise to [-1, 1]
    peak = np.abs(waveform).max()
    if peak > 0:
        waveform = waveform / peak

    model_path = _resolve_path(model_choice)
    try:
        processor, model, device = _load(str(model_path))
    except Exception as exc:
        return "", f"✗ Failed to load model: {exc}"

    input_features = processor(
        waveform,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
    ).input_features.to(device)

    generate_kwargs: dict = {"num_beams": 5, "max_length": 448}
    lang_code = LANGUAGES.get(language_label)
    if lang_code:
        generate_kwargs["language"] = lang_code

    with torch.no_grad():
        predicted_ids = model.generate(input_features, **generate_kwargs)

    text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
    status = f"✅  Model: {model_choice}  |  Language hint: {language_label}  |  Device: {device.upper()}"
    return text, status


# ---------------------------------------------------------------------------
# Side-by-side comparison helper
# ---------------------------------------------------------------------------

def compare(audio, model_a: str, model_b: str, language_label: str):
    text_a, status_a = transcribe(audio, model_a, language_label)
    text_b, status_b = transcribe(audio, model_b, language_label)
    return text_a, status_a, text_b, status_b


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Whisper HF Model Tester") as demo:

        gr.Markdown(
            """
            # 🎙️ Whisper Nigerian Language — HF Model Tester
            Test the fine-tuned `whisper-small-nigerian` model (Yoruba · Hausa · Igbo · Pidgin).
            The **★ best** entry in the dropdown is the checkpoint with the lowest WER.
            The **[final]** model always contains the best checkpoint weights.
            Upload an audio file **or** record from your microphone, then hit **Transcribe**.
            """
        )

        # ---- Single model tab ----
        with gr.Tab("Single Model"):
            with gr.Row():
                audio_in = gr.Audio(
                    sources=["microphone", "upload"],
                    type="numpy",
                    label="Audio Input",
                )
            with gr.Row():
                model_dd = gr.Dropdown(
                    choices=MODEL_CHOICES,
                    value=BEST_MODEL_DEFAULT or (MODEL_CHOICES[0] if MODEL_CHOICES else None),
                    label="Model / Checkpoint",
                )
                lang_dd = gr.Dropdown(
                    choices=list(LANGUAGES.keys()),
                    value="Auto-detect",
                    label="Language hint",
                )
            btn = gr.Button("Transcribe", variant="primary")
            output_text = gr.Textbox(label="Transcription", lines=4, show_copy_button=True)
            status_text = gr.Textbox(label="Status", interactive=False, lines=1)

            btn.click(
                fn=transcribe,
                inputs=[audio_in, model_dd, lang_dd],
                outputs=[output_text, status_text],
            )

        # ---- Side-by-side comparison tab ----
        with gr.Tab("Compare Two Models"):
            gr.Markdown("Upload the **same audio** and compare two model outputs side by side.")
            with gr.Row():
                cmp_audio = gr.Audio(
                    sources=["microphone", "upload"],
                    type="numpy",
                    label="Audio Input",
                )
                cmp_lang = gr.Dropdown(
                    choices=list(LANGUAGES.keys()),
                    value="Auto-detect",
                    label="Language hint",
                )
            with gr.Row():
                cmp_model_a = gr.Dropdown(
                    choices=MODEL_CHOICES,
                    value=BEST_MODEL_DEFAULT or (MODEL_CHOICES[0] if len(MODEL_CHOICES) > 0 else None),
                    label="Model A",
                )
                cmp_model_b = gr.Dropdown(
                    choices=MODEL_CHOICES,
                    value=MODEL_CHOICES[1] if len(MODEL_CHOICES) > 1 else None,
                    label="Model B",
                )
            cmp_btn = gr.Button("Compare", variant="primary")
            with gr.Row():
                out_a = gr.Textbox(label="Model A — Transcription", lines=4, show_copy_button=True)
                out_b = gr.Textbox(label="Model B — Transcription", lines=4, show_copy_button=True)
            with gr.Row():
                stat_a = gr.Textbox(label="Model A — Status", interactive=False)
                stat_b = gr.Textbox(label="Model B — Status", interactive=False)

            cmp_btn.click(
                fn=compare,
                inputs=[cmp_audio, cmp_model_a, cmp_model_b, cmp_lang],
                outputs=[out_a, stat_a, out_b, stat_b],
            )

        # ---- Model info tab ----
        with gr.Tab("Model Info"):
            info_lines = []
            for label, root in MODEL_ROOTS.items():
                if not root.exists():
                    info_lines.append(f"**{label}**: ❌ folder not found at `{root}`")
                    continue
                best_info = _load_best_info(root)
                ckpts = sorted(
                    [d.name for d in root.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
                    key=lambda n: int(n.split("-")[1]),
                )
                best_step = best_info.get("best_step")
                best_wer  = best_info.get("best_metric_wer")
                ckpt_display = ", ".join(
                    f"{c} ★" if best_step and c == f"checkpoint-{best_step}" else c
                    for c in ckpts
                ) if ckpts else "none"
                info_lines.append(
                    f"**{label}**  \n"
                    f"Path: `{root}`  \n"
                    f"Checkpoints: {ckpt_display}  \n"
                    f"Final model: {'✅' if (root/'model.safetensors').exists() else '❌'}  \n"
                    + (f"Best checkpoint: `checkpoint-{best_step}` — WER {best_wer:.2f}%  \n"
                       f"*(final model contains best weights)*"
                       if best_step else "")
                )
            gr.Markdown("\n\n---\n\n".join(info_lines))

    return demo


if __name__ == "__main__":
    demo = build_ui()
demo.launch(share=False, inbrowser=True, theme=gr.themes.Soft())
