from models.Logistic_Regression.logistic_regression_numpy import LogisticRegressionNumpy
import numpy as np

class LogisticRegressionOVR:
    def __init__(self, lr=0.01, epochs=100, batch_size=None, patience=5):
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.models = {}
        self.classes_ = []

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        for cls in self.classes_:
            print(f"\nTraining classifier for class: {cls}")
            binary_y = (y == cls).astype(int)
            model = LogisticRegressionNumpy(lr=self.lr, epochs=self.epochs, batch_size=self.batch_size, patience=self.patience)
            model.fit(X, binary_y)
            self.models[cls] = model

    def predict(self, X):
        probs = np.zeros((X.shape[0], len(self.classes_)))
        for i, cls in enumerate(self.classes_):
            probs[:, i] = self.models[cls].predict_proba(X)
        return np.array(self.classes_)[np.argmax(probs, axis=1)]
