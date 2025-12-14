import streamlit as st
import os
import re
import tempfile
import librosa
import numpy as np
import soundfile as sf
import noisereduce as nr
import torch
from faster_whisper import WhisperModel
# PERUBAHAN KRITIS: spellchecker dipasang sebagai pyspellchecker
from spellchecker import SpellChecker 
from rapidfuzz import process, fuzz
from rapidfuzz.distance import Levenshtein
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# --- A. MODEL INITIALIZATION (CACHED) ---

@st.cache_resource
def load_whisper_model():
    # Mengubah model dari "large-v3" menjadi "small.en"
    print("Loading WhisperModel (small.en)…")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" and torch.cuda.is_available() else "int8"
    
    # PERUBAHAN DI SINI: menggunakan model yang lebih kecil
    model = WhisperModel("small.en", device=device, compute_type=compute_type) 
    print(f"✔ WhisperModel loaded. Running on: {device.upper()}")
    return model

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

WHISPER_MODEL = load_whisper_model()
EMBEDDING_MODEL = load_embedding_model()

# --- B. CONSTANTS & RESOURCES ---

spell = SpellChecker(language="en")
ML_TERMS = [
    "tensorflow", "keras", "vgc16", "vgc19", "mobilenet",
    "efficientnet", "cnn", "relu", "dropout", "model",
    "layer normalization", "batch normalization", "attention",
    "embedding", "deep learning", "dataset", "submission"
]
PHRASE_MAP = {
    "celiac" : "cellular", "script" : "skripsi", "i mentioned" : "submission",
    "time short flow": "tensorflow", "eras": "keras", "vic": "vgc16",
    "vic": "vgc19", "va": "vgc16", "va": "vgc19", "mobile net": "mobilenet",
    "data set" : "dataset", "violation laws" : "validation loss"
}

# --- C. PRE-PROCESSING & TEXT CLEANING FUNCTIONS ---

def apply_noise_reduction_and_normalize(y, sr, prop_decrease=0.6):
    y_clean = nr.reduce_noise(y=y, sr=sr, prop_decrease=prop_decrease)
    if np.max(np.abs(y_clean)) > 0:
        y_normalized = y_clean / np.max(np.abs(y_clean)) * 0.98
    else:
        y_normalized = y_clean
    return y_normalized

def correct_ml_terms(word):
    w = word.lower()
    if w in spell.word_frequency.words():
        return word
    match, score, _ = process.extractOne(w, ML_TERMS)
    dist = Levenshtein.distance(w, match.lower())
    if dist <= 3 or score >= 65:
        return match
    return word

def fix_context_outliers(text):
    words = text.split()
    if len(words) < 3:
        return text
    try:
        word_embeds = EMBEDDING_MODEL.encode(words)
        sent_embed = EMBEDDING_MODEL.encode([text])[0]
        sims = cosine_similarity(word_embeds, [sent_embed]).flatten()
        outlier_idx = sims.argmin()
        match, score, _ = process.extractOne(words[outlier_idx], words)
        if score < 95:
            words[outlier_idx] = match
    except Exception:
        pass
    return " ".join(words)

def remove_duplicate_words(text):
    words = text.split()
    res = []
    for i, w in enumerate(words):
        if i == 0 or w != words[i - 1]:
            res.append(w)
    return " ".join(res)

def clean_text(text, use_embedding_fix=True):
    # Logika Text Cleaning
    text = text.lower()
    fillers = ["umm", "uh", "uhh", "erm", "hmm", "eee", "emmm", "yeah", "ah", "okay", "vic"]
    pattern = r"\b(" + "|".join(fillers) + r")\b"
    text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    text = re.sub(r"\.{2,}", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    for wrong, correct in PHRASE_MAP.items():
        text = re.sub(rf"\b{re.escape(wrong)}\b", correct, text)
    words = []
    for w in text.split():
        sp = spell.correction(w)
        if sp:
            w = sp
        w = correct_ml_terms(w)
        words.append(w)
    text = " ".join(words)
    if use_embedding_fix:
        text = fix_context_outliers(text)
    text = remove_duplicate_words(text)
    return text.capitalize()

# --- D. MAIN TRANSCRIPTION FUNCTION ---

def get_transcription_data(temp_file_path):
    """
    Handles audio pre-processing, transcription, and text cleaning.
    """
    
    try:
        y, sr = librosa.load(temp_file_path, sr=16000)
        
        # Simpan audio yang sudah diproses ke file sementara baru
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as processed_tmp:
            y_processed = apply_noise_reduction_and_normalize(y, sr)
            sf.write(processed_tmp.name, y_processed, sr)
            processed_tmp_path = processed_tmp.name

            # TRANSCRIPTION
            segments, info = WHISPER_MODEL.transcribe(
                processed_tmp_path, 
                language="en", 
                task="transcribe", 
                temperature=0, 
                beam_size=4, 
                vad_filter=True
            )
            raw_transcript = " ".join([seg.text for seg in segments]).strip()
        
        # TEXT CLEANING
        clean_transcript = clean_text(raw_transcript, use_embedding_fix=True)

        return {
            "raw_transcript": raw_transcript,
            "clean_transcript": clean_transcript,
            "success": True
        }

    except Exception as e:
        return {
            "raw_transcript": "",
            "clean_transcript": "",
            "error": f"STT Processing failed: {str(e)}",
            "success": False
        }

if __name__ == '__main__':
    print("STT Module ready to be imported by streamlit_app.py.")
