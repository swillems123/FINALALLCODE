import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# Download VADER lexicon if not already downloaded
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except nltk.downloader.DownloadError:
    nltk.download('vader_lexicon')

df = pd.read_csv("tube_english_only_with_topics.csv")
sid = SentimentIntensityAnalyzer()

# Calculate sentiment score for each hook
df['hook_sentiment'] = df['hook'].apply(lambda x: sid.polarity_scores(str(x))['compound'])

# Check if the hook contains a question mark
df['hook_has_question'] = df['hook'].str.contains(r'\?', regex=True)

print("--- Hook Sentiment Analysis ---")
print("This section shows the average view count for videos grouped by the sentiment score of their hook (first 50 words).")
print("Sentiment scores range from -1 (very negative) to +1 (very positive). We group them into 5 bins:")
# Calculate and print average views grouped by sentiment bins
sentiment_groups = df.groupby(pd.cut(df['hook_sentiment'], bins=5))['viewCount'].mean()
print(sentiment_groups)
print("\nInterpretation: This helps see if more positive or negative hooks tend to get more views on average.")

print("\n--- Question in Hook Analysis ---")
print("This section shows the average view count for videos based on whether their hook (first 50 words) contains a question mark.")
# Calculate and print average views grouped by whether the hook has a question
question_groups = df.groupby('hook_has_question')['viewCount'].mean()
print(question_groups)
print("\nInterpretation: 'True' means the hook contains a question, 'False' means it does not. This helps see if asking a question early correlates with average views.")

# Optional: Print correlation coefficients for a more direct measure
sentiment_corr = df['hook_sentiment'].corr(df['viewCount'])
question_corr = df['hook_has_question'].astype(int).corr(df['viewCount'])

print(f"\nDirect correlation between hook sentiment and view count: {sentiment_corr:.3f}")
print(f"Direct correlation between having a question in the hook and view count: {question_corr:.3f}")
print("(Correlation values close to 0 indicate a weak relationship, close to 1 or -1 indicate a stronger relationship).")
