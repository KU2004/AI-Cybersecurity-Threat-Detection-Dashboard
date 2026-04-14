from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os

def train_model(X):
    model = IsolationForest(contamination=0.2, random_state=42)
    model.fit(X)

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/model.pkl")

    return model

def evaluate_model(model, X, y):
    preds = model.predict(X)

    preds = [1 if p == -1 else 0 for p in preds]

    print("\nConfusion Matrix:")
    print(confusion_matrix(y, preds))

    print("\nClassification Report:")
    print(classification_report(y, preds))

    return preds