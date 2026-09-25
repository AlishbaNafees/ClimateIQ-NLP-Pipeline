# CLIMATEIQ HYBRID PIPELINE
# ============================================================
# PIPELINE COMBINED — Orchestrates all 3 NLP Layers
# ============================================================
# INPUT:  data/raw/climate_tweets.csv
# OUTPUT:
#   data/processed/layer1_cleaned.csv      (after Layer 1)
#   data/processed/layer2_sentiment.csv    (after Layer 2)
#   data/final/climate_tweets_final.csv    (after Layer 3)
# ============================================================
import pandas as pd
from pathlib import Path
import logging
from tqdm import tqdm
 
from pipeline.layer1_cleaning      import run_cleaning
from pipeline.layer2_vader_scoring import run_vader_scoring
from pipeline.layer3_rule_based    import run_rule_based
 
# ============================================================
# LOGGING SETUP
# ============================================================
 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
 
# ============================================================
# PATH CONFIGURATION
# ============================================================
 
INPUT_PATH  = Path("data/raw/climate_tweets.csv")
 
LAYER1_OUT  = Path("data/processed/layer1_cleaned.csv")
LAYER2_OUT  = Path("data/processed/layer2_sentiment.csv")
FINAL_OUT   = Path("data/final/climate_tweets_final.csv")
 
ROWS_TO_LOAD = 2500000   # 25 Lakh Rows
 
# ============================================================
# MAIN PIPELINE FUNCTION
# ============================================================
def run_pipeline():
 
    print("\n" + "="*55)
    print("  CLIMATEIQ NLP PIPELINE — ClimateIQ v5")
    print(" All 3 Layers")
    print("="*55)
 
    # ----------------------------------------------------------
    # LOAD RAW DATA
    # ----------------------------------------------------------
    try:
        logger.info(f"Loading dataset: {INPUT_PATH}")
        df = pd.read_csv(INPUT_PATH, nrows=ROWS_TO_LOAD)
        logger.info(f"Dataset loaded — {len(df):,} rows, {len(df.columns)} columns")
    except FileNotFoundError:
        logger.error(f"File not found: {INPUT_PATH}")
        return
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        return
 
    # ----------------------------------------------------------
    # LAYER 1 — Data Cleaning 
    # ----------------------------------------------------------
    try:
        df = run_cleaning(df)
        LAYER1_OUT.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(LAYER1_OUT, index=False)
        logger.info(f"Layer 1 output saved → {LAYER1_OUT}")
    except Exception as e:
        logger.error(f"Layer 1 failed: {e}")
        return
 
    # ----------------------------------------------------------
    # LAYER 2 — Lexical Sentiment Analysis 
    # ----------------------------------------------------------
    try:
        df = run_vader_scoring(df)
        LAYER2_OUT.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(LAYER2_OUT, index=False)
        logger.info(f"Layer 2 output saved → {LAYER2_OUT}")
    except Exception as e:
        logger.error(f"Layer 2 failed: {e}")
        return
 
    # ----------------------------------------------------------
    # LAYER 3 — Rule-Based Classification 
    # ----------------------------------------------------------
    try:
        df = run_rule_based(df)
        FINAL_OUT.parent.mkdir(parents=True, exist_ok=True)
        CHUNK_SIZE = 100000
        for i, start in enumerate(range(0, len(df), CHUNK_SIZE)):
            chunk = df.iloc[start:start + CHUNK_SIZE]
            chunk.to_csv(FINAL_OUT, mode='w' if i == 0 else 'a',
                         index=False, header=(i == 0))
        logger.info(f"Layer 3 output saved → {FINAL_OUT}")
    except Exception as e:
        logger.error(f"Layer 3 failed: {e}")
        return
 
    # ----------------------------------------------------------
    # PIPELINE COMPLETE
    # ----------------------------------------------------------
    print("\n" + "="*55)
    print("  ✅ PIPELINE COMPLETE!")
    print("="*55)
    print(f"  Total rows processed  : {len(df):,}")
    print(f"  Total columns (final) : {len(df.columns)}")
    print(f"\n  Output files saved:")
    print(f"  → {LAYER1_OUT}")
    print(f"  → {LAYER2_OUT}")
    print(f"  → {FINAL_OUT}")
    print("="*55 + "\n")