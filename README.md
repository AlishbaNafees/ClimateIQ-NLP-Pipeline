# ClimateIQ NLP Pipeline

A three-layer pipeline that cleans, scores and classifies 2.5 million climate-related tweets from 47 cities. This is the NLP layer behind the ClimateIQ_v5 desktop app. That app reads whatever this pipeline produces.

## What it does

**Layer 1 cleans the text.** Tweets get lowercased, tokenized, stripped of stopwords and punctuation, and lemmatized with NLTK. URLs and emojis are pulled out into their own columns for reference, though they stay in the cleaned text itself.

**Layer 2 scores sentiment.** Each cleaned tweet gets a TextBlob polarity score and a VADER compound score, combined into one number, 40% TextBlob and 60% VADER. VADER carries the heavier weight because it reads informal, slang-heavy text more reliably than TextBlob does alone. Scores above 0.12 are labeled positive, below -0.12 negative, everything else neutral. This layer also tags an emotion category by keyword match, things like Disaster Risk or Health Hazard.

**Layer 3 classifies by topic.** A keyword classifier checks the cleaned text against the sentiment score from Layer 2 and assigns a topic (Disaster, Pollution, Resource Issue, Climate Change, or General) along with a predicted emotion.

## Output files

| File | What it holds |
|---|---|
| `data/processed/layer1_cleaned.csv` | Output after text cleaning |
| `data/processed/layer2_sentiment.csv` | Output after sentiment scoring |
| `data/final/climate_tweets_final.csv` | Final classified output |

## Running it

1. Place `climate_tweets.csv` inside `data/raw/`.
2. Install dependencies: `pip install -r requirements.txt`
3. Run the pipeline: `python main.py`

## A note on the dataset

The raw and processed CSVs aren't in this repository. At 2.5 million rows, they're well past what GitHub is meant to hold, so this repo carries the code, not the data.

## Limitations

Sentiment is scored after stopword removal, and NLTK's stopword list includes words like "no" and "not." That means a tweet like "not good" gets scored as "good," since the negation is already gone by the time Layer 2 sees the text. Fixing this means moving the cleaning order around so negations survive into scoring, and that's the next thing worth doing here. Labels also haven't been checked against hand-labeled tweets, so treat them as a first pass, not ground truth.

## Related work

The dataset this pipeline produces feeds into [ClimateIQ_v5](https://github.com/AlishbaNafees/ClimateIQ_v5), a PyQt6 desktop app that turns it into charts and PDF reports. That app was built by a team of three. My part was this NLP pipeline, plus a fix on the application side: the app's summary and reports were only showing 50,000 tweets out of 2.5 million, and I traced it to a hardcoded row limit in the data manager, raised it, and confirmed the true record count with a SQL query.

Built as part of a three-person final year project. NLP pipeline design and implementation by Alishba Nafees.
