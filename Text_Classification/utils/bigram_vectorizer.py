from utils.extract_bigrams import extract_bigrams
from typing import List
import numpy as np

class BigramVectorizer:
    def __init__(self):
        self.vocab = []
        self.bigram_to_idx = {}
    
    def build_vocab(self, tokenized_texts: List[List[str]]):
        """
        Build a vocabulary (set of unique bigrams) from tokenized sentences.
        """
        vocab_set = set()
        for tokens in tokenized_texts:
            bigrams = extract_bigrams(tokens)
            vocab_set.update(bigrams)

        self.vocab = sorted(list(vocab_set))
        self.bigram_to_idx = {bg: i for i, bg in enumerate(self.vocab)}

    def transform(self, tokenized_texts: List[List[str]]) -> np.ndarray:
        """
        Convert tokenized sentences into binary feature vectors
        based on presence/absence of bigrams.
        """
        X = np.zeros((len(tokenized_texts), len(self.vocab)), dtype=np.float32)

        for i, tokens in enumerate(tokenized_texts):
            bigrams = extract_bigrams(tokens)
            for bg in bigrams:
                if bg in self.bigram_to_idx:
                    X[i, self.bigram_to_idx[bg]] = 1
        return X
    
    def fit_transform(self, tokenized_texts: List[List[str]]) -> np.ndarray:
        """
        Build vocabulary and return the transformed feature matrix.
        """
        self.build_vocab(tokenized_texts)
        return self.transform(tokenized_texts)


