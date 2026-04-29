import os
from datasets import load_dataset, Audio
p = "C:/Users/jerem/Documents/Projects/Whisper-Ng/datasets/cmn1qen4q00xjo107gln14ztz/cv-corpus-25.0-2026-03-09/ha"
tsv_path = os.path.join(p, "train.tsv")
clips_dir = os.path.join(p, "clips")
ds = load_dataset("csv", data_files=tsv_path, delimiter="\t", split="train")

def _add_audio_path(batch):
    batch["audio"] = os.path.join(clips_dir, str(batch["path"]))
    batch["text"] = str(batch["sentence"])
    return batch
ds = ds.map(_add_audio_path)
ds = ds.cast_column("audio", Audio(sampling_rate=16000))
print(ds[0])
