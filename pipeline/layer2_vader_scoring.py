# CLIMATEIQ HYBRID PIPELINE
# ============================================================
# LAYER 2: Lexical Sentiment Analysis
# TextBlob + VADER Hybrid Scoring
# ============================================================
# INPUT:  Cleaned dataframe from Layer 1 (via pipeline_combined.py)
# OUTPUT: Same dataframe with 3 columns FILLED:
#         Sentiment_Score | Sentiment_Label | Emotion_Type
# ============================================================
 
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
 
# ============================================================
# CONFIGURATION
# ============================================================
 
POS_THRESHOLD = 0.12
NEG_THRESHOLD = -0.12
 
TEXTBLOB_WEIGHT = 0.4
VADER_WEIGHT    = 0.6
 
# ============================================================
# EMOTION KEYWORDS — Dictionary Based
# ============================================================
 
EMOTION_KEYWORDS = {
    "Resource Crisis"  : ["drought", "water scarcity"],
    "Disaster Risk"    : ["flood", "heavy rainfall"],
    "Climate Stress"   : ["heatwave", "rising temperature"],
    "Health Hazard"    : ["air pollution", "smog"],
    "Climate Concern"  : ["climate change", "extreme weather"],
    "Economic Impact"  : ["economic pressure"],
    "Safety Threat"    : ["public safety issue"],
    "Long Term Risk"   : ["long-term effect"]
}
 
# ============================================================
# VADER — initialize once 
# ============================================================
 
vader_analyzer = SentimentIntensityAnalyzer()
 
# ============================================================
# CORE SENTIMENT FUNCTION
# ============================================================
 
def calculate_lexical_sentiment(text):
    """
    Calculates hybrid sentiment using TextBlob + VADER.
    Detects emotion category via keyword dictionary.
 
    Args:
        text (str): Cleaned tweet text
 
    Returns:
        tuple: (Sentiment_Score, Sentiment_Label, Emotion_Type)
    """
    if not isinstance(text, str) or text.strip() == "":
        return 0.0, "Neutral", "Unknown"
 
    # TextBlob polarity
    textblob_score = TextBlob(text).sentiment.polarity
 
    # VADER compound score
    vader_score = vader_analyzer.polarity_scores(text)["compound"]
 
    # Weighted hybrid score
    final_score = (TEXTBLOB_WEIGHT * textblob_score) + \
                  (VADER_WEIGHT    * vader_score)
 
    # Sentiment Label
    if final_score > POS_THRESHOLD:
        label = "Positive"
    elif final_score < NEG_THRESHOLD:
        label = "Negative"
    else:
        label = "Neutral"
 
    # Emotion Detection
    emotion_type = "General Climate Discussion"
    for emotion, keywords in EMOTION_KEYWORDS.items():
        if any(keyword in text for keyword in keywords):
            emotion_type = emotion
            break
 
    return final_score, label, emotion_type
 
# ============================================================
# MAIN SCORING FUNCTION — called by pipeline_combined.py
# ============================================================
 
def run_vader_scoring(df):
    """
    Fills Sentiment_Score, Sentiment_Label, Emotion_Type columns.
    These columns already exist in raw dataset but were empty.
    Called by pipeline_combined.py
 
    Args:
        df (pd.DataFrame): Cleaned dataframe from Layer 1
 
    Returns:
        pd.DataFrame: Dataframe with 3 sentiment columns filled
    """
    print("\n" + "="*55)
    print("  LAYER 2 — Lexical Sentiment Analysis")
    print("  (TextBlob + VADER Hybrid Scoring)")
    print("="*55)
 
    from tqdm import tqdm
    tqdm.pandas(desc="Scoring tweets")
 
    results = df["Cleaned_Text"].progress_apply(
        lambda x: pd.Series(calculate_lexical_sentiment(x))
    )
 
    # Fill the 3 originally-empty columns
    df["Sentiment_Score"] = results[0]
    df["Sentiment_Label"] = results[1]
    df["Emotion_Type"]    = results[2]
 
    print("✅ Sentiment_Score filled")
    print("✅ Sentiment_Label filled  (Positive / Negative / Neutral)")
    print("✅ Emotion_Type    filled  (Dictionary-based detection)")
 
    # Distribution summary
    counts = df["Sentiment_Label"].value_counts()
    print(f"\n   Sentiment Distribution:")
    for label, count in counts.items():
        print(f"   {label:10} → {count:,} tweets")
 
    print("\n✅ LAYER 2 COMPLETE")
    print(f"   Rows processed : {len(df)}")
    print(f"   Columns now    : {len(df.columns)}")
 
    return df