"""Data preprocessing

Module for text preprocessing functions, including tokenization,
lowercasing, and removing special characters.
"""

import re
from nltk.tokenize import WordPunctTokenizer


def preprocess_text(text):
    """Preprocess the input text by lowercasing and removing special characters,
    and tokenizing the text into words.

    Args:
        text (list): The input text to preprocess.

    Returns:
        list: A list of preprocessed texts.
    """
    preprocessed_texts = []
    tokenizer = WordPunctTokenizer()
    for t in text:
        # Lowercase the text
        t = t.lower()
        # Remove special characters
        t = re.sub(r"[^a-zA-Z0-9-'\s]", "", t)
        # Tokenize the text
        preprocessed_texts.append(tokenizer.tokenize(t))
    return preprocessed_texts


if __name__ == "__main__":
    import pandas as pd

    df = pd.read_parquet("Text_Classification/data/train-00000-of-00001.parquet")
    text = df["text"].tolist()
    labels = df["label"].tolist()

    tokenized_text = preprocess_text(text)

    print("Sample texts:")
    for i in range(3):
        print(f"Text {i+1}: {text[i]}")
        print(f"Tokenized Text {i+1}: {tokenized_text[i]}")
