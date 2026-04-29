# Experiment Overview: Fine-Tuning Whisper for Nigerian Languages

## Objective
The primary objective of this research is to adapt the `openai/whisper-small` speech-to-text model for Nigerian languages (Hausa, Yoruba, Igbo, and Nigerian Pidgin) using Parameter-Efficient Fine-Tuning (PEFT) and Low-Rank Adaptation (LoRA) in 8-bit precision.

## Architecture & Framework
- **Base Model:** `openai/whisper-small`
- **Tuning Method:** LoRA (Low-Rank Adaptation) via Hugging Face `peft`
- **Precision:** 8-bit quantization (`BitsAndBytesConfig`)
- **Evaluation Framework:** Normalized Word Error Rate (WER) across combined local (Common Voice) and remote datasets (e.g., `google/WaxalNLP`).

## Key Focus Areas
1. **Robust Evaluation:** Overcoming artificially high baseline error rates caused by rigid test dataset orthography.
2. **Data Efficiency:** Maximizing the utility of scarce local dataset resources by algorithmically rebalancing train/val/test splits.
3. **Training Dynamics:** Identifying the optimum point of convergence before catastrophic overfitting occurs in low-resource settings.
