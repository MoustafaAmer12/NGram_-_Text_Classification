
from nltk.corpus import stopwords
import string
from nltk.tokenize import word_tokenize
from typing import List
import re

class TextPreprocessor:
    def __init__(self, remove_stopwords: bool = True, language: str = 'english'):
        self.remove_stopwords = remove_stopwords
        if remove_stopwords:
            self.stop_words = set(stopwords.words(language))
        else:
            self.stop_words = set()

    def normalize_text(self, text: str) -> str:
        """
        Lowercase, remove punctuation and special characters.
        """
        text = text.lower()
        text = re.sub(f"[{string.punctuation}]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize the text into words.
        """
        tokens = word_tokenize(text)
        if self.remove_stopwords:
            tokens = [t for t in tokens if t not in self.stop_words]
        return tokens
    def preprocess(self, text: str) -> List[str]:
        """
        Apply full preprocessing pipeline.
        """
        text = self.normalize_text(text)
        tokens = self.tokenize(text)
        return tokens
