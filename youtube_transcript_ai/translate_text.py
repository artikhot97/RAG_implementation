from langdetect import detect
from transformers import MarianMTModel, MarianTokenizer

model_name = "Helsinki-NLP/opus-mt-hi-en"
model = MarianMTModel.from_pretrained(model_name)
tokenizer = MarianTokenizer.from_pretrained(model_name)

def translate_hindi_to_english(text):
    if detect(text) == "hi":
        tokens = tokenizer([text], return_tensors="pt", padding=True, truncation=True)
        translated = model.generate(**tokens)
        translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
        return translated_text
    else:
        return text