import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

# Download NLTK stopwords (only needed the first time)
nltk.download("stopwords")

# Load the dataset (update the path as needed)
df = pd.read_csv("Tweets.csv")

# Print original column names (for debugging)
print(df.columns)

# Load English stopwords
stop_words = set(stopwords.words("english"))

# Text cleaning function (uses regular expression tokenization)
def clean_text(text):
    text = str(text)  # Convert to string to avoid NaN issues
    text = re.sub(r"http\S+", "", text)       # Remove URLs
    text = re.sub(r"@\w+", "", text)          # Remove @mentions
    text = re.sub(r"[^a-zA-Z\s]", "", text)   # Remove punctuation and numbers
    text = text.lower()                       # Convert to lowercase
    tokens = re.findall(r"\b[a-z]+\b", text)  # Tokenize using regex
    filtered = [w for w in tokens if w not in stop_words]  # Remove stopwords
    return " ".join(filtered)

# Apply the cleaning function
df["clean_text"] = df["text"].apply(clean_text)

# Output a sample of cleaned results
print(df[["text", "clean_text"]].sample(5))

# Optionally save the cleaned data to a new file
df.to_csv("cleaned_tweets.csv", index=False)
print("Cleaned data saved as 'cleaned_tweets.csv'")
