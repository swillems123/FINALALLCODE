"""
analysis_pipeline.py

A unified pipeline to analyze YouTube transcripts for:
1. Word-frequency correlation with a performance metric.
2. Sentiment analysis correlation.
3. Topic modeling performance by topic.
4. Structural features (questions in intro) correlation.

Outputs results to the console and saves CSV summaries in the working directory.
"""

import pandas as pd
import numpy as np
import nltk
from scipy.stats import pearsonr
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# --- Hardcoded settings ---
INPUT_CSV = "tube_english_only_with_metadata.csv"
METRIC = "viewCount"  # Change to "likeCount" or "commentCount" if desired
TOP_WORDS = 20
NUM_TOPICS = 8

def main():
    df = pd.read_csv(INPUT_CSV)
    # Ensure 'text' and metric columns exist
    if 'text' not in df.columns:
        raise ValueError("CSV must contain a 'text' column.")
    if METRIC not in df.columns:
        raise ValueError(f"CSV must contain the metric column '{METRIC}'.")

    # Convert metric to numeric (in case it's read as string)
    df[METRIC] = pd.to_numeric(df[METRIC], errors='coerce').fillna(0)

    # --- Word frequency correlation ---
    vectorizer = CountVectorizer(stop_words='english', max_features=TOP_WORDS)
    X = vectorizer.fit_transform(df['text'].fillna(''))
    word_freq = np.asarray(X.sum(axis=0)).flatten()
    words = vectorizer.get_feature_names_out()
    metric_values = df[METRIC].values

    correlations = []
    for i, word in enumerate(words):
        word_counts = X[:, i].toarray().flatten()
        corr, _ = pearsonr(word_counts, metric_values)
        correlations.append((word, corr))
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)
    print("Top word correlations with", METRIC)
    for word, corr in correlations:
        print(f"{word}: {corr:.3f}")

    # --- Sentiment analysis correlation ---
    nltk.download('vader_lexicon')
    sid = SentimentIntensityAnalyzer()
    df['sentiment'] = df['text'].fillna('').apply(lambda x: sid.polarity_scores(x)['compound'])
    sentiment_corr, _ = pearsonr(df['sentiment'], metric_values)
    print(f"\nSentiment correlation with {METRIC}: {sentiment_corr:.3f}")

    # --- Topic modeling ---
    tfidf = TfidfVectorizer(stop_words='english', max_features=1000)
    X_tfidf = tfidf.fit_transform(df['text'].fillna(''))
    lda = LatentDirichletAllocation(n_components=NUM_TOPICS, random_state=0)
    lda_topics = lda.fit_transform(X_tfidf)
    df['topic'] = np.argmax(lda_topics, axis=1)
    topic_means = df.groupby('topic')[METRIC].mean()
    print("\nAverage metric by topic:")
    print(topic_means)

    # --- Structural feature (question in intro) ---
    df['has_question_intro'] = df['text'].fillna('').str[:200].str.contains(r'\?', regex=True)
    question_corr, _ = pearsonr(df['has_question_intro'].astype(int), metric_values)
    print(f"\nCorrelation of question in intro with {METRIC}: {question_corr:.3f}")

    # Save results
    df.to_csv("analysis_pipeline_output.csv", index=False)
    print("\nAnalysis complete. Results saved to analysis_pipeline_output.csv")

if __name__ == "__main__":
    main()