ClimateIQ Hybrid Pipeline

A 3-layer Hybrid pipeline for processing 25 lakh climate tweets.


How to Run


1. Place climate_tweets.csv inside data/raw/
2. Install dependencies:


   pip install -r requirements.txt


3. Run the pipeline:


   python main.py


Output Files

File                                    
data/processed/layer1_cleaned.csv     (After text cleaning) 
data/processed/layer2_sentiment.csv   (After sentiment scoring)
data/final/climate_tweets_final.csv   (Final classified output)


Pipeline Layers

layer1_cleaning.py                    (Regex + NLTKLayer)
layer2_vader_scoring.py               (TextBlob + VADER HybridLayer) 
layer3_rule_based.py                  (Keyword + Rule-Based)