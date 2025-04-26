"""
Standalone YouTube Transcript Topic Analysis

- Loads a CSV with YouTube transcripts and metadata.
- Runs LDA topic modeling and prints top words for each topic.
- Calculates correlation between topic assignment and view count.
- Adds a 'hook' column (first 50 words of each transcript).
- Saves enhanced DataFrame to a new CSV.
- Prints summary statistics for each topic.
- Prints descriptions to explain each result.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from scipy.stats import pearsonr

# --- Settings ---
INPUT_CSV = "tube_english_only_with_metadata.csv"
OUTPUT_CSV = "tube_english_only_with_topics.csv"
TEXT_COL = "text"
METRIC = "viewCount"
NUM_TOPICS = 8
N_TOP_WORDS = 12

def main():
    print("Loading data...")
    df = pd.read_csv(INPUT_CSV)
    if TEXT_COL not in df.columns:
        raise ValueError(f"Column '{TEXT_COL}' not found in CSV.")
    if METRIC not in df.columns:
        raise ValueError(f"Metric column '{METRIC}' not found in CSV.")

    # Ensure metric is numeric
    df[METRIC] = pd.to_numeric(df[METRIC], errors='coerce').fillna(0)

    print("Vectorizing transcripts...")
    tfidf = TfidfVectorizer(stop_words='english', max_features=2000)
    X_tfidf = tfidf.fit_transform(df[TEXT_COL].fillna(''))

    print(f"Running LDA topic modeling with {NUM_TOPICS} topics...")
    lda = LatentDirichletAllocation(n_components=NUM_TOPICS, random_state=0)
    lda_topics = lda.fit_transform(X_tfidf)
    df['topic'] = np.argmax(lda_topics, axis=1)

    # Print top words for each topic
    feature_names = tfidf.get_feature_names_out()
    print("\nTop words for each topic:")
    print("These are the most representative words for each topic, helping you interpret what each topic is about.")
    for topic_idx, topic in enumerate(lda.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-N_TOP_WORDS - 1:-1]]
        print(f"Topic {topic_idx}: {' | '.join(top_words)}")

    # Correlation between topic assignment and view count
    print("\nCorrelation between topic assignment and view count:")
    print("This shows how strongly being assigned to a topic is associated with higher or lower view counts. Positive means videos in this topic tend to have more views, negative means fewer views.")
    metric_values = df[METRIC].values
    for topic_num in range(NUM_TOPICS):
        topic_mask = (df['topic'] == topic_num)
        corr, _ = pearsonr(topic_mask.astype(int), metric_values)
        print(f"Topic {topic_num}: correlation with {METRIC} = {corr:.3f}")

    # Add hook (first 50 words) column
    def get_hook(text):
        words = str(text).split()
        return ' '.join(words[:50])
    df['hook'] = df[TEXT_COL].apply(get_hook)

    # Print summary statistics for each topic
    print("\nSummary statistics by topic:")
    print("These are the average view, like, and comment counts for videos in each topic.")
    summary = df.groupby('topic')[[METRIC, 'likeCount', 'commentCount']].mean().round(2)
    print(summary)

    print("\nThe 'hook' column (first 50 words of each transcript) has been added to the output CSV. This can help you analyze how the opening of each video relates to its performance.")

    # Save enhanced DataFrame
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nAnalysis complete. Results saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()