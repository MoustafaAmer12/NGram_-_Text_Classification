"""Naive Bayes text classification model.

This module implements a Naive Bayes classifier for text classification tasks from scratch.
It includes methods for training the model and making predictions.
"""

import numpy as np


class NaiveBayes:
    """Naive Bayes Manually Implemented"""

    def __init__(self, X, y, alpha=1):
        self.X = X
        self.y = y

        self.classes, self.counts = np.unique(self.y, return_counts=True)
        self.class_to_index = {c: i for i, c in enumerate(self.classes)}

        self.bow = None
        self.bow_mapper = None

        self.class_priors = None

        self.alpha = alpha
        self.class_word_counts = None
        self.word_likelihoods = None

    def _compute_priors(self):
        """_summary_"""
        self.class_priors = self.counts / self.counts.sum()

        self.class_priors = np.log(self.class_priors).reshape(1, -1)

    def _construct_bow(self):
        """_summary_"""
        self.bow = set()

        for doc in self.X:
            for token in doc:
                self.bow.add(token)

        # Convert the BoW to Key-Value Pairs
        self.bow_mapper = {word: i for i, word in enumerate(self.bow)}

    def _compute_word_counts(self):
        """_summary_"""
        self._construct_bow()

        # The alpha here is for the laplacian smoothing
        self.class_word_counts = np.zeros((len(self.bow), len(self.classes)))

        for doc, label in zip(self.X, self.y):
            class_idx = self.class_to_index[label]
            for token in doc:
                self.class_word_counts[self.bow_mapper[token]][class_idx] += 1

    def _compute_word_likelihoods(self):
        """_summary_"""
        self._compute_word_counts()

        col_sums = self.class_word_counts.sum(axis=0, keepdims=True)
        col_sums += self.alpha * len(self.bow)

        class_words_with_alpha = self.class_word_counts + self.alpha

        self.word_likelihoods = np.log(class_words_with_alpha) - np.log(col_sums)

    def train(self):
        self._compute_priors()
        self._compute_word_likelihoods()

    def test(self, X):
        """_summary_

        Args:
            x (_type_): _description_
        """
        y_pred = []

        for doc in X:
            ids = np.array([], dtype=int)
            for token in doc:
                ids = np.append(ids, self.bow_mapper.get(token, -1))

            ids = ids[ids != -1]
            feats = self.word_likelihoods[ids]
            likelihood = feats.sum(axis=0, keepdims=True)

            likelihood = likelihood + self.class_priors

            y_pred.append(int(np.argmax(likelihood)))

        return y_pred


if __name__ == "__main__":
    import pandas as pd
    from Text_Classification.utils.preprocessing import preprocess_text
    from Text_Classification.utils.eval_metrics import compare_metrics
    from sklearn.model_selection import train_test_split

    df = pd.read_parquet("Text_Classification/data/train-00000-of-00001.parquet")

    text = df["text"].tolist()
    labels = df["label"].tolist()

    tokenized_text = preprocess_text(text)

    X_train, X_dev, y_train, y_dev = train_test_split(
        tokenized_text, labels, test_size=0.2, random_state=42
    )

    classifier = NaiveBayes(X_train, y_train, alpha=1)

    classifier.train()

    y_pred_train = classifier.test(X_train)
    y_pred_dev = classifier.test(X_dev)

    compare_metrics(y_train, y_pred_train, task="Train")
    compare_metrics(y_dev, y_pred_dev, task="Eval")
