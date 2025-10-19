"""Naive Bayes sckl text classification model"""

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


class NaiveBayesSKL:
    def __init__(self, X, y):
        self.X = X
        self.y = y

        self.vectorizer = CountVectorizer()
        self.model = MultinomialNB()

    def train(self):
        self.X_train_counts = self.vectorizer.fit_transform(self.X)

        self.model.fit(self.X_train_counts, self.y)

    def test(self, X):
        X_test_counts = self.vectorizer.transform(X)

        return self.model.predict(X_test_counts)


if __name__ == "__main__":
    import pandas as pd
    from Text_Classification.utils.eval_metrics import compare_metrics
    from sklearn.model_selection import train_test_split

    df = pd.read_parquet("Text_Classification/data/train-00000-of-00001.parquet")

    text = df["text"].tolist()
    labels = df["label"].tolist()

    X_train, X_dev, y_train, y_dev = train_test_split(
        text, labels, test_size=0.2, random_state=42
    )

    classifier = NaiveBayesSKL(X_train, y_train)

    classifier.train()

    y_pred_train = classifier.test(X_train)
    y_pred_dev = classifier.test(X_dev)

    compare_metrics(y_train, y_pred_train, "Train")
    compare_metrics(y_dev, y_pred_dev, "Eval")
