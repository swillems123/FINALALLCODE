import pandas as pd
from langdetect import detect, LangDetectException
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

csv_file = "tube_english_only.csv"
df = pd.read_csv(csv_file)

def detect_lang(text):
    try:
        return detect(str(text))
    except LangDetectException:
        return None

# Use ThreadPoolExecutor with a set number of workers (e.g., 8)
with ThreadPoolExecutor(max_workers=8) as executor:
    detected = list(tqdm(executor.map(detect_lang, df['text']), total=len(df)))

df['detected_language'] = detected

# Filter to only English texts
df_en = df[df['detected_language'] == 'en']

df_en.to_csv("tube_english_only_detected.csv", index=False)