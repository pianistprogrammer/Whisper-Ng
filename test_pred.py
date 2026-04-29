import torch
import unicodedata
from transformers import WhisperProcessor, WhisperForConditionalGeneration, BitsAndBytesConfig
from peft import PeftModel
from datasets import load_dataset, Audio

model_id = './whisper-small-nigerian-lora'
base_id = 'openai/whisper-small'

processor = WhisperProcessor.from_pretrained(base_id, task='transcribe')
model = WhisperForConditionalGeneration.from_pretrained(base_id, quantization_config=BitsAndBytesConfig(load_in_8bit=True), device_map='auto')
model = PeftModel.from_pretrained(model, model_id)
model.eval()

ds = load_dataset('google/WaxalNLP', 'yor_tts', split='test').cast_column('audio', Audio(sampling_rate=16000))
sample = ds[0]
audio = sample['audio']
text = sample['text']

input_features = processor.feature_extractor(audio['array'], sampling_rate=audio['sampling_rate'], return_tensors='pt').input_features.to('cuda')
forced_ids = processor.get_decoder_prompt_ids(language='yoruba', task='transcribe')
    
with torch.no_grad():
    out = model.generate(input_features.to('cuda'), forced_decoder_ids=forced_ids, max_new_tokens=255)
pred = processor.tokenizer.batch_decode(out, skip_special_tokens=True)[0]

print('REF:', text)
print('PRED:', pred)
