# CLIMATEIQ HYBRID PIPELINE
# ============================================================
# LAYER 3: Rule-Based Classification
# Keyword + Sentiment Score Based Classifier
# ============================================================
# INPUT:  Dataframe from Layer 2 (via pipeline_combined.py)
# OUTPUT: Same dataframe with 2 new columns added:
#         Predicted_Emotion | Predicted_Topic
# ============================================================
 
import pandas as pd
 
# ============================================================
# KEYWORD DICTIONARIES
# ============================================================
 
DISASTER_KEYWORDS = [
    "flood", "flooding", "heavy rainfall", "river overflow",
    "drought", "cyclone", "storm", "heatwave"
]
 
POLLUTION_KEYWORDS = [
    "smog", "air pollution", "pollution", "toxic air",
    "carbon emission", "greenhouse gases"
]
 
RESOURCE_KEYWORDS = [
    "water scarcity", "water shortage", "food crisis",
    "crop failure"
]
 
TEMPERATURE_KEYWORDS = [
    "rising temperature", "global warming",
    "climate change", "temperature rise"
]
 
NEGATION_WORDS = ["no", "not", "never", "without"]
 
# ============================================================
# CORE CLASSIFIER FUNCTION
# ============================================================
 
def classify_rule_based(text, sentiment_score):
    """
    Classifies tweet using keyword matching + sentiment score.
 
    Args:
        text (str):            Cleaned tweet text
        sentiment_score (float): Hybrid score from Layer 2
 
    Returns:
        tuple: (Predicted_Emotion, Predicted_Topic)
    """
    text = str(text).lower()
    negation_present = any(neg in text for neg in NEGATION_WORDS)
 
    if any(word in text for word in DISASTER_KEYWORDS):
        if sentiment_score < 0:
            return "Severe Disaster Concern", "Disaster"
        elif negation_present:
            return "Disaster Relief / Recovery", "Disaster"
        else:
            return "General Disaster Discussion", "Disaster"
 
    elif any(word in text for word in POLLUTION_KEYWORDS):
        if sentiment_score < 0:
            return "Health Hazard", "Pollution"
        else:
            return "Environmental Awareness", "Pollution"
 
    elif any(word in text for word in RESOURCE_KEYWORDS):
        if sentiment_score < 0:
            return "Resource Crisis", "Resource Issue"
        else:
            return "Resource Discussion", "Resource Issue"
 
    elif any(word in text for word in TEMPERATURE_KEYWORDS):
        if sentiment_score < 0:
            return "Climate Stress", "Climate Change"
        else:
            return "Climate Awareness", "Climate Change"
 
    else:
        if sentiment_score > 0:
            return "Positive Environmental Outlook", "General"
        elif sentiment_score < 0:
            return "Negative Climate Concern", "General"
        else:
            return "Neutral Discussion", "General"
 
# ============================================================
# MAIN CLASSIFICATION FUNCTION — called by pipeline_combined.py
# ============================================================
 
def run_rule_based(df):
    """
    Applies rule-based classification to the dataframe.
    Called by pipeline_combined.py
 
    Args:
        df (pd.DataFrame): Dataframe from Layer 2
 
    Returns:
        pd.DataFrame: Dataframe with 2 new columns added
    """
    print("\n" + "="*55)
    print("  LAYER 3 — Rule-Based Classification")
    print("  (Keyword Matching + Sentiment Score)")
    print("="*55)
 
    from tqdm import tqdm
    tqdm.pandas(desc="Classifying tweets")
 
    df[["Predicted_Emotion", "Predicted_Topic"]] = df.progress_apply(
        lambda row: pd.Series(
            classify_rule_based(row["Cleaned_Text"], row["Sentiment_Score"])
        ),
        axis=1
    )
 
    print("✅ Predicted_Emotion column added")
    print("✅ Predicted_Topic   column added")
 
    # Topic distribution summary
    counts = df["Predicted_Topic"].value_counts()
    print(f"\n   Topic Distribution:")
    for topic, count in counts.items():
        print(f"   {topic:25} → {count:,} tweets")
 
    print("\n✅ LAYER 3 COMPLETE")
    print(f"   Rows processed : {len(df)}")
    print(f"   Columns now    : {len(df.columns)}")
 
    return df