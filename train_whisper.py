import librosa
import numpy as np
from datasets import load_dataset
from transformers import WhisperTokenizer
import mlx.core as mx
import mlx.optimizers as optim
import mlx_whisper
from mlx_lm import lora
import mlx_lm

# -----------------------------
# Config
# -----------------------------

MODEL_NAME = "base"
DATASET_NAME = "google/WaxalNLP"
DATASET_CONFIG = "yor_tts"

SAMPLE_RATE = 16000
BATCH_SIZE = 8
EPOCHS = 5
LEARNING_RATE = 1e-4

OUTPUT_ADAPTER = "yoruba_whisper_lora"


# -----------------------------
# Feature extraction
# -----------------------------

def compute_features(audio_array):

    mel = librosa.feature.melspectrogram(
        y=audio_array,
        sr=SAMPLE_RATE,
        n_mels=80,
        fmax=8000
    )

    log_mel = librosa.power_to_db(mel)

    return log_mel.astype(np.float32)


# -----------------------------
# Dataset preprocessing
# -----------------------------

def preprocess(example):

    audio = example["audio"]["array"]

    features = compute_features(audio)

    return {
        "input_features": features,
        "text": example["text"]
    }


def tokenize(example):

    example["labels"] = tokenizer(example["text"]).input_ids
    return example


# -----------------------------
# Load dataset
# -----------------------------

print("Loading Waxal Yoruba dataset...")

dataset = load_dataset(DATASET_NAME, DATASET_CONFIG)

train_ds = dataset["train"]
val_ds = dataset["validation"]

train_ds = train_ds.map(preprocess)
val_ds = val_ds.map(preprocess)

# -----------------------------
# Tokenizer
# -----------------------------

print("Loading tokenizer...")

tokenizer = WhisperTokenizer.from_pretrained(
    "openai/whisper-base",
    language="yoruba",
    task="transcribe"
)

train_ds = train_ds.map(tokenize)
val_ds = val_ds.map(tokenize)


# -----------------------------
# Load Whisper MLX model
# -----------------------------

print("Loading Whisper MLX model...")

model = mlx_whisper.load_model(MODEL_NAME)

# -----------------------------
# Apply LoRA
# -----------------------------

print("Applying LoRA adapters...")

lora_config = {
    "r": 16,
    "alpha": 32,
    "dropout": 0.05,
    "target_modules": ["query", "value", "key", "out"]
}

model = lora.apply_lora(model, lora_config)


# -----------------------------
# Optimizer
# -----------------------------

optimizer = optim.AdamW(learning_rate=LEARNING_RATE)


# -----------------------------
# Batch helper
# -----------------------------

def get_batches(dataset, batch_size):

    batch = []

    for item in dataset:

        batch.append(item)

        if len(batch) == batch_size:
            yield batch
            batch = []

    if batch:
        yield batch


# -----------------------------
# Training loss
# -----------------------------

def loss_fn(model, batch):

    features = mx.array([x["input_features"] for x in batch])
    labels = mx.array([x["labels"] for x in batch])

    logits = model(features)

    loss = mx.mean(
        mx.softmax_cross_entropy(
            logits,
            labels
        )
    )

    return loss


# -----------------------------
# Training loop
# -----------------------------

print("Starting training...")

for epoch in range(EPOCHS):

    losses = []

    for batch in get_batches(train_ds, BATCH_SIZE):

        loss, grads = mx.value_and_grad(loss_fn)(model, batch)

        optimizer.update(model, grads)

        losses.append(loss.item())

    avg_loss = sum(losses) / len(losses)

    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.4f}")


# -----------------------------
# Save adapter
# -----------------------------

print("Saving LoRA adapter...")

mlx_lm.save_adapter(model, OUTPUT_ADAPTER)

print("Training complete!")
print("Adapter saved to:", OUTPUT_ADAPTER)