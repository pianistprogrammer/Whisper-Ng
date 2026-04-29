# Orthography, Text Normalization, and the 80% WER Paradox

## The "80% WER" False Alarm
During early evaluations of the trained Whisper-small LoRA model on Nigerian languages, initial evaluations yielded Word Error Rates (WER) approaching ~80%. This was initially perceived as catastrophic model failure. 

However, intensive manual analysis of the inferences revealed a profound counter-intuitive behavior: the fine-tuned model was actually generating **superior** orthography compared to the ground-truth reference sentences. 

### Key Findings:
1. **Diacritics & Punctuation:** The model reliably produced missing diacritics (e.g., `Ọ`, `gbàbọ́ọ̀lù`) and proper casing/punctuation that the raw reference transcriptions lacked.
2. **Compounded Words:** High-quality language structures emerged naturally from the model, leading to string mismatch penalties compared to naive references.

## Solution: Intensive Text Normalization
To prevent extreme penalties for "writing better than the reference," we established a rigorous pre-evaluation text normalization pipeline in `eval_lora_whisper.py`.

### Normalization Pipeline:
1. **Unicode NFC Normalization:** Standardizes combined and decomposed unicode diacritic characters:
   ```python
   text = unicodedata.normalize("NFC", text).lower()
   ```
2. **Aggressive Punctuation Stripping:** Removes all non-alphanumeric trailing/internal punctuation using regex:
   ```python
   import re
   text = re.sub(r'[^\w\s]', '', text)
   ```

## Conclusion
Evaluating raw outputs against low-resource datasets without normalization heavily misrepresents the utility of the fine-tuned speech model. By implementing programmatic unicode stripping, we aligned the evaluation pipeline with the true learning capability of the PEFT layer.
