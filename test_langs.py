from transformers import WhisperTokenizer
tokenizer = WhisperTokenizer.from_pretrained('openai/whisper-small')
try:
    print(tokenizer.get_decoder_prompt_ids(language='igbo', task='transcribe'))
    print("igbo is supported")
except Exception as e:
    print(e)
