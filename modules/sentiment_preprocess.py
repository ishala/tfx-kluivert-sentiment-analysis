import pandas as pd
import re
import os
import string
import json
import csv
import requests
import nltk
from io import StringIO
from sklearn.preprocessing import LabelEncoder
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')

# Inisiasi Variabel Path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))  # Ambil root project
DATA_ROOT = os.path.join(BASE_DIR, 'data')  # Pastikan path selalu ke 'data'

PATH_DATA_RAW = os.path.join(DATA_ROOT, 'raw_comments.csv')  # Path ke CSV

PATH_PROCESSED_DATA = os.path.join(BASE_DIR, 'data', 'youtube-comments', 'processed_comments.csv')


# Fungsi untuk Memuat Lexicon
def load_lexicon():
    lexicon_positive, lexicon_negative = {}, {}

    response = requests.get('https://raw.githubusercontent.com/angelmetanosaa/dataset/main/lexicon_positive.csv')
    if response.status_code == 200:
        reader = csv.reader(StringIO(response.text), delimiter=',')
        lexicon_positive = {row[0]: int(row[1]) for row in reader}
    else:
        print("❌ Failed to fetch positive lexicon data")

    response = requests.get('https://raw.githubusercontent.com/angelmetanosaa/dataset/main/lexicon_negative.csv')
    if response.status_code == 200:
        reader = csv.reader(StringIO(response.text), delimiter=',')
        lexicon_negative = {row[0]: int(row[1]) for row in reader}
    else:
        print("❌ Failed to fetch negative lexicon data")

    return lexicon_positive, lexicon_negative


# Load lexicon saat script pertama kali dijalankan
lexicon_positive, lexicon_negative = load_lexicon()

# Fungsi-fungsi Preprocessing
def cleaning_data(df):
    return df.dropna(axis=0)

def delete_duplicate(df):
    return df.drop_duplicates()

def cleaning_text(text):
    if not isinstance(text, str):
        return ""  # Jika bukan string, return string kosong

    text = re.sub(r'@[A-Za-z0-9]+', '', text)  # Menghapus mention
    text = re.sub(r'#[A-Za-z0-9]+', '', text)  # Menghapus hashtag
    text = re.sub(r"http\S+", '', text)  # Menghapus link
    text = re.sub(r'[0-9]+', '', text)  # Menghapus angka
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Menghapus karakter non-alphabet
    text = text.translate(str.maketrans('', '', string.punctuation))  # Menghapus tanda baca
    text = text.strip()  # Trim whitespace
    return text

def case_folding(text):
    return text.lower() if isinstance(text, str) else text

def replace_slang_words(text, slangDict):
    words = text.split()
    cleaned_words = [slangDict.get(word.lower(), word) for word in words]
    return ' '.join(cleaned_words)

def tokenize_word(text):
    return word_tokenize(text) if isinstance(text, str) else []

def stopword_removal(text):
    stopwords_id = set(stopwords.words('indonesian'))
    stopwords_en = set(stopwords.words('english'))
    stopwords_all = stopwords_id.union(stopwords_en)

    return [word for word in text if word not in stopwords_all]

def sentiment_analysis_lexicon_indonesia(text):
    score = 0
    for word in text:
        if word in lexicon_positive:
            score += lexicon_positive[word]
        if word in lexicon_negative:
            score -= lexicon_negative[word]  # Negatif harus dikurangkan!
    if score >= 1:
        polarity = 'positive'
    elif score <= -1:
        polarity = 'negative'
    else:
        polarity = 'neutral'

    return score, polarity

def to_sentence(words_list):
    return ' '.join(words_list) if isinstance(words_list, list) else ""

# Fungsi utama untuk menjalankan preprocessing
def preprocess_and_save(df, text_column='comment', label='polarity'):
    if text_column not in df.columns:
        print(f"❌ Kolom '{text_column}' tidak ditemukan dalam dataset!")
        return None

    df = df.iloc[0:5000]
    df = df.dropna(subset=[text_column])
    df[text_column] = df[text_column].astype(str)  # Pastikan semua nilai diubah jadi string

    df[text_column] = df[text_column].apply(cleaning_text)
    df[text_column] = df[text_column].apply(case_folding)
    df[text_column] = df[text_column].apply(tokenize_word)
    df[text_column] = df[text_column].apply(stopword_removal)

    # Sentiment Analysis
    results = df[text_column].apply(sentiment_analysis_lexicon_indonesia)
    df['polarity_score'], df['polarity'] = zip(*results)

    # Konversi kembali ke bentuk kalimat
    df[text_column] = df[text_column].apply(to_sentence)

    # Konversi kategori polarity ke numerik
    label_encoder = LabelEncoder()
    df[label] = label_encoder.fit_transform(df[label])
    # ambil hanya komentar dan label
    df = df[[text_column, label]]
    
    # Simpan hasil preprocessing
    os.makedirs(os.path.dirname(PATH_PROCESSED_DATA), exist_ok=True)
    
    df.to_csv(PATH_PROCESSED_DATA, index=False)
    print(f"✅ Data tersimpan di {PATH_PROCESSED_DATA}")

    return df


# Load Data
if os.path.exists(PATH_DATA_RAW):
    df = pd.read_csv(PATH_DATA_RAW)
else:
    print(f"❌ File {PATH_DATA_RAW} tidak ditemukan!")
    df = None

if __name__ == "__main__":
    if df is not None:
        preprocess_and_save(df)