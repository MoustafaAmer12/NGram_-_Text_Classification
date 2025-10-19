from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from datasets import load_dataset
from utils.text_preprocessor import TextPreprocessor
from utils.bigram_vectorizer import BigramVectorizer
from models.Logistic_Regression.logistic_regression_ovr import LogisticRegressionOVR
import numpy as np

dataset = load_dataset("avsolatorio/mteb-emotion-avs_triplets", split="train")

# 1. Preprocess text
preprocessor = TextPreprocessor(remove_stopwords=True)
texts = [preprocessor.preprocess(sample['text']) for sample in dataset]

# 2. Extract labels
labels = np.array([sample['label'] for sample in dataset])

# 3. Build bigram features
vectorizer = BigramVectorizer()
X = vectorizer.fit_transform(texts)

# 4. Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42, stratify=labels)

# 5. Train multi-class Logistic Regression
model_ovr = LogisticRegressionOVR(lr=0.1, epochs=10, batch_size=32, patience=5)
model_ovr.fit(X_train, y_train)

# 6. Evaluate
y_pred = model_ovr.predict(X_test)
print("\nOverall Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
