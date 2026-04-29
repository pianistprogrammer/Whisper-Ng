# Fine-Tuning Observations, Overfitting, and Early Stopping

## Training Loss & Convergence Behaviors
Over multiple iteration runs, fine-tuning `openai/whisper-small` across combined Nigerian data via LoRA layers revealed distinct convergence thresholds regarding validation loss and overfitting.

### The Overfitting Horizon
1. **Eval Loss Minimum:** Minimum validation scalar loss was cleanly observed at **Epoch 4**, producing the sweet spot of model convergence (`eval_loss` absolute minimum around `0.952`).
2. **Post-Epoch 4 Divergence:** By Epochs 6-8, validation tracking decoupled entirely from the training loss. Rapid and exponential overfitting spikes were detected explicitly via `trainer_state.json`.

## Early Stopping
Given that the `peft` LoRA layers aggressively memorize low-resource dialects within a few full iterations over the datasets, strict Early Stopping monitors were validated as essential criteria. 
* By incorporating Early Stopping based directly on `eval_loss` against the newly configured `15%` validation data (documented in our split distribution strategy), the system prevents the model from mapping the raw dataset noise and rigidly retaining only the targeted phonetics.

### Recommended Safeguards
* Enable Early Stopping explicitly on the `eval_loss` metric.
* Avoid configurations exceeding 4-5 epochs over local low-resource African languages unless significant regularizations (e.g. dropout scaling, dynamic augmentations) are introduced.
