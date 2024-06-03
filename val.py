import json
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os
import gradio as gr # type: ignore
import signal
import sys
import google.generativeai as genai # type: ignore
from dotenv import load_dotenv
from fuzzywuzzy import fuzz # type: ignore

print("my hub link :")
print("https://huggingface.co/ihebaker10")
print("essai d'install avec pip install -r requirements.txt")

# Load environment variables
GOOGLE_API_KEY = "............" #change by your api key ! 
genai.configure(api_key=GOOGLE_API_KEY)
model1 = genai.GenerativeModel('gemini-pro')

model_mapping = {
    "es": "ihebaker10/spark-name-es-to-en",
    "ar": "ihebaker10/spark-name-ar-to-en",
    "ka": "ihebaker10/spark-name-ka-to-en",
    "hy": "ihebaker10/spark-name-hy-to-en",
    "de": "ihebaker10/spark-name-de-to-en",
    "my": "ihebaker10/spark-name-my-to-en",
    "ru": "ihebaker10/spark-name-ru-to-en",
    "hi": "ihebaker10/spark-name-hi-to-en",
    "ja": "ihebaker10/spark-name-ja-to-en",
    "zh": "ihebaker10/spark-name-zh-to-en",
    "ko": "ihebaker10/spark-name-ko-to-en",
    "fr": "ihebaker10/spark-name-fr-to-en",
}


error_data_count = {lang: 0 for lang in model_mapping.keys()}

# Translation function for each model
def translate(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = model.generate(**inputs)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Translation function by LLM with updated prompt
def translate_text(text, source_lang):
    prompt = f"""You are a highly skilled linguist and an expert in translating personal names from various languages into English. Your task is to provide the most accurate and culturally appropriate translation of the following name from {source_lang} to English: {text}.

Please adhere to the following guidelines:
- Provide only the translated name without any additional text or examples.
- Ensure the translation maintains the phonetic and cultural integrity of the original name.
- Refer to the following language codes for your translations:
  - hy = Armenian
  - ka = Georgian
  - ja = Japanese
  - es = Spanish
  - hi = Hindi
  - ar = Arabic
  - fr = French
  - my = Burmese
  - de = German
  - ru = Russian
  - zh = Chinese

Example:
  - Translating from Arabic to English:
    - Input: أحمد محمد
    - Output: Ahmed Mohamed

Translate the following name from {source_lang} to English: {text}
"""
    response = model1.generate_content(prompt)
    translated = response.text
    return translated.strip()

# New similarity function using fuzzywuzzy
def fuzzywuzzy_similarity(str1, str2):
    return fuzz.ratio(str1, str2) / 100

def calculate_similarity(text1, text2):
    return fuzzywuzzy_similarity(text1, text2)

def store_error_data(source_lang, source_text, correct_translation):
    global error_data_count
    error_data_file = f'error_data_{source_lang}.jsonl'
    new_entry = {"translation": {source_lang: source_text, "en": correct_translation}}
    
    # Check for duplicates
    if os.path.exists(error_data_file):
        with open(error_data_file, 'r') as file:
            existing_data = [json.loads(line) for line in file]
            if new_entry in existing_data:
                print(f"Duplicate data found, skipping storage: {new_entry}")
                return

    with open(error_data_file, 'a') as file:
        json.dump(new_entry, file)
        file.write('\n')
        error_data_count[source_lang] += 1
        print(f"Stored error data for {source_lang}. Total count: {error_data_count[source_lang]}")

def retrain_models():
    for lang, model_checkpoint in model_mapping.items():
        error_data_file = f'error_data_{lang}.jsonl'
        if os.path.exists(error_data_file):
            with open(error_data_file, 'r') as file:
                lines = file.readlines()
                if len(lines) >= 5: # changer  5 -> 100 
                    print(f"Triggering retraining for {lang}...")
                    file.close()  #  closed 
                    os.system(f"python train.py {model_checkpoint} {lang} {error_data_file}")
                    if os.path.exists(error_data_file):
                        os.remove(error_data_file)  
                    error_data_count[lang] = 0  # Reset 
                else:
                    print(f"Not enough data to retrain for {lang}. Current count: {len(lines)}")

def translate_with_model(source_lang, source_text):
    global tokenizer, model, model_checkpoint
    model_checkpoint = model_mapping[source_lang]
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
    
    predicted = translate(source_text)
    print("predicted text model: ", predicted)
    correct_translation = translate_text(source_text, source_lang)
    print("correct translation model: ", correct_translation)
    similarity = calculate_similarity(predicted, correct_translation)
    print(f"Similarity score: {similarity}")
    if similarity < 0.98:
        store_error_data(source_lang, source_text, correct_translation)
    return correct_translation if similarity < 0.98 else predicted

def gradio_interface(source_lang, source_text):
    return translate_with_model(source_lang, source_text)

def exit_handler(signum, frame):
    # Retrain models for all languages upon exit
    retrain_models()
    print("Exiting and retraining complete.")
    sys.exit(0)  # Ensure the program exits

# Set up signal handler for exit
signal.signal(signal.SIGINT, exit_handler)
signal.signal(signal.SIGTERM, exit_handler)

iface = gr.Interface(
    fn=gradio_interface,
    inputs=[gr.Dropdown(choices=list(model_mapping.keys()), label="Source Language"), gr.Textbox(label="Text to Translate")],
    outputs="text",
    title="Translation Model"
)

try:
    iface.launch(share=True)
except Exception as e:
    print(f"An error occurred: {e}")

# Ensure retraining when the user exits the interface
import atexit
atexit.register(retrain_models)
