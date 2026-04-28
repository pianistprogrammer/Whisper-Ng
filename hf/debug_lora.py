import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
from datasets import load_dataset, Audio

PEFT_MODEL = "./whisper-small-nigerian-lora"
peft_config = PeftConfig.from_pretrained(PEFT_MODEL)
base_model_name = peft_config.base_model_name_or_path
processor = WhisperProcessor.from_pretrained(base_model_name)
model = WhisperForConditionalGeneration.from_pretrained(
    base_model_name, 
    quantization_config=BitsAndBytesConfig(load_in_8bit=True), 
    device_map="auto"
)
model = PeftModel.from_pretrained(model, PEFT_MODEL)
model.eval()
model.config.use_cache = True
if hasattr(model.generation_config, "forced_decoder_ids"):
    model.generation_config.forced_decoder_ids = None 
if hasattr(model.generation_config, "suppress_tokens"):
    model.generation_config.suppress_tokens = []

ds = load_dataset("google/WaxalNLP", "yor_tts", split="test")
ds = ds.cast_column("audio", Audio(sampling_rate=16000))
sample = ds[0]

inputs = processor(sample["audio"]["array"], sampling_rate=16000, return_tensors="pt").input_features.to("cuda")

with torch.cuda.amp.autocast():
    with torch.no_grad():
        out = model.generate(inputs, max_new_tokens=255)
        text = processor.batch_decode(out, skip_special_tokens=True)[0]
        raw_tokens = processor.batch_decode(out, skip_special_tokens=False)[0]

print("\n--- RESULTS ---")
print("REFERENCE:", sample["text"])
print("PREDICTION:", text)
print("RAW TOKENS:", raw_tokens)
