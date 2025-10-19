import numpy as np
from typing import Optional

class LogisticRegressionNumpy:
    def __init__(
        self,
        lr: float = 0.01,
        epochs: int = 100,
        batch_size: Optional[int] = None,
        seed: int = 42,
        early_stopping: bool = True,
        patience: int = 5,
        min_delta: float = 1e-4
    ):
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed
        self.early_stopping = early_stopping
        self.patience = patience
        self.min_delta = min_delta

        self.W = None
        self.b = None

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def _cross_entropy_loss(self, y_true, y_pred):
        eps = 1e-10
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def _shuffle_data(self, X, y):
        np.random.seed(self.seed)
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        return X[indices], y[indices]

    def fit(self, X: np.ndarray, y: np.ndarray):
        n_samples, n_features = X.shape
        self.W = np.zeros(n_features)
        self.b = 0

        best_loss = float('inf')
        no_improve_count = 0

        for epoch in range(self.epochs):
            X, y = self._shuffle_data(X, y)

            if self.batch_size is None:
                X_batches = [X]
                y_batches = [y]
            else:
                X_batches = [X[i:i+self.batch_size] for i in range(0, n_samples, self.batch_size)]
                y_batches = [y[i:i+self.batch_size] for i in range(0, n_samples, self.batch_size)]

            total_loss = 0

            for X_batch, y_batch in zip(X_batches, y_batches):
                z = np.dot(X_batch, self.W) + self.b
                y_pred = self._sigmoid(z)
                loss = self._cross_entropy_loss(y_batch, y_pred)
                total_loss += loss

                dw = np.dot(X_batch.T, (y_pred - y_batch)) / len(y_batch)
                db = np.mean(y_pred - y_batch)

                self.W -= self.lr * dw
                self.b -= self.lr * db

            avg_loss = total_loss / len(X_batches)
            print(f"Epoch {epoch+1}/{self.epochs} - Loss: {avg_loss:.6f}")

            # ----- Early stopping check -----
            if self.early_stopping:
                if best_loss - avg_loss > self.min_delta:
                    best_loss = avg_loss
                    no_improve_count = 0
                else:
                    no_improve_count += 1

                if no_improve_count >= self.patience:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break
            # ---------------------------------

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        z = np.dot(X, self.W) + self.b
        return self._sigmoid(z)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)
