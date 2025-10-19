import nltk
import re

# Download required NLTK data
nltk.download("gutenberg")
nltk.download("punkt")

from nltk.corpus import gutenberg

print("Libraries imported and data downloaded successfully!")


def preprocess_text(text, remove_stopwords=False):
    """
    Preprocess text by:
    - Converting to lowercase
    - Removing punctuation and special characters
    - Tokenizing into words
    - Optionally removing stopwords
    """
    # Convert to lowercase
    text = text.lower()

    # Remove punctuation and special characters (keep only letters and spaces)
    text = re.sub(r"[^a-z\s]", "", text)

    # Tokenize by splitting on whitespace
    tokens = text.split()

    # Remove stopwords if specified
    if remove_stopwords:
        from nltk.corpus import stopwords

        nltk.download("stopwords")
        stop_words = set(stopwords.words("english"))
        tokens = [word for word in tokens if word not in stop_words]

    # Remove empty strings
    tokens = [token for token in tokens if token]

    return tokens


def load_data(remove_stopwords=False):
    train_text1 = gutenberg.raw("shakespeare-caesar.txt")
    train_text2 = gutenberg.raw("shakespeare-macbeth.txt")
    train_text = train_text1 + " " + train_text2

    eval_text = gutenberg.raw("shakespeare-hamlet.txt")

    train_tokens = preprocess_text(train_text)
    eval_tokens = preprocess_text(eval_text)

    return train_tokens, eval_tokens


if __name__ == "__main__":
    train_tokens, eval_tokens = load_data()
