# CLIMATEIQ HYBRID PIPELINE
# ============================================================
# LAYER 1: Data Cleaning
# Regex + NLTK Preprocessing
# ============================================================
# INPUT:  data/raw/climate_tweets.csv
# OUTPUT: No output (functions only — called by pipeline_combined.py)
# ============================================================

import re
import string
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# --- Download required NLTK resources ---
nltk.download("punkt",       quiet=True)
nltk.download("punkt_tab",   quiet=True)
nltk.download("stopwords",   quiet=True)
nltk.download("wordnet",     quiet=True)

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def extract_urls(text):
    """Extract all URLs from tweet text."""
    if pd.isna(text):
        return []
    return re.findall(r'https?://\S+|www\.\S+', text)


def extract_emojis(text):
    """Extract all emojis from tweet text."""
    if pd.isna(text):
        return []
    return re.findall(
        r'[\U0001F600-\U0001F64F'
        r'\U0001F300-\U0001F5FF'
        r'\U0001F680-\U0001F6FF'
        r'\U0001F1E0-\U0001F1FF]+', text)


def avg_word_length(text):
    """Calculate average word length of tweet text."""
    if pd.isna(text):
        return 0
    words = text.split()
    if len(words) == 0:
        return 0
    return sum(len(w) for w in words) / len(words)


# ============================================================
# MAIN CLEANING FUNCTION
# ============================================================

def run_cleaning(df):
    """
    Applies all Layer 1 cleaning steps to the dataframe.
    Called by pipeline_combined.py
    
    Args:
        df (pd.DataFrame): Raw tweets dataframe
    
    Returns:
        pd.DataFrame: Cleaned dataframe with new feature columns
    """
    print("\n" + "="*55)
    print("  LAYER 1 — Data Cleaning (Regex + NLTK)")
    print("="*55)

    # Ensure string type
    df["Tweet_Text"] = df["Tweet_Text"].astype(str)

    # Step 1 — Lowercase
    df["Lowercase_Text"] = df["Tweet_Text"].str.lower()
    print("✅ Step 1 — Text Normalized (Lowercase)")

    # Step 2 — Extract URLs
    df["URLs_Found"] = df["Tweet_Text"].apply(extract_urls)
    print("✅ Step 2 — URLs Extracted")

    # Step 3 — Extract Emojis
    df["Emojis"] = df["Tweet_Text"].apply(extract_emojis)
    print("✅ Step 3 — Emojis Extracted")

    # Step 4 — Detect Questions & Exclamations
    df["Contains_Question"]    = df["Tweet_Text"].str.contains(r'\?', regex=True)
    df["Contains_Exclamation"] = df["Tweet_Text"].str.contains(r'!',  regex=True)
    print("✅ Step 4 — Questions & Exclamations Detected")

    # Step 5 — Average Word Length
    df["Avg_Word_Length"] = df["Tweet_Text"].apply(avg_word_length)
    print("✅ Step 5 — Average Word Length Calculated")

    # Step 6 — Tokenization
    df["Tokens"] = df["Lowercase_Text"].apply(word_tokenize)
    print("✅ Step 6 — Tokenization Done")

    # Step 7 — Remove Stopwords
    stop_words = set(stopwords.words("english"))
    df["Tokens_No_Stopwords"] = df["Tokens"].apply(
        lambda words: [w for w in words if w not in stop_words]
    )
    print("✅ Step 7 — Stopwords Removed")

    # Step 8 — Remove Punctuation
    df["Tokens_No_Punct"] = df["Tokens_No_Stopwords"].apply(
        lambda words: [w for w in words if w not in string.punctuation]
    )
    print("✅ Step 8 — Punctuation Removed")

    # Step 9 — Lemmatization
    lemmatizer = WordNetLemmatizer()
    df["Lemmatized_Tokens"] = df["Tokens_No_Punct"].apply(
        lambda words: [lemmatizer.lemmatize(w) for w in words]
    )
    print("✅ Step 9 — Lemmatization Done")

    # Step 10 — Cleaned Text (tokens → string)
    df["Cleaned_Text"] = df["Lemmatized_Tokens"].apply(
        lambda words: " ".join(words)
    )
    print("✅ Step 10 — Cleaned Text Column Created")

    # Step 11 — Word Count Features
    df["Word_Count"]        = df["Cleaned_Text"].apply(lambda x: len(x.split()))
    df["Unique_Word_Count"] = df["Cleaned_Text"].apply(lambda x: len(set(x.split())))
    print("✅ Step 11 — Word Count & Unique Word Count Added")

    print("\n✅ LAYER 1 COMPLETE")
    print(f"   Rows processed : {len(df)}")
    print(f"   Columns now    : {len(df.columns)}")

    return df