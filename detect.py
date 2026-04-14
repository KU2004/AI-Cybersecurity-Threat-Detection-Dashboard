import joblib

def detect_threat(data):
    model = joblib.load("models/model.pkl")
    preds = model.predict(data)

    alerts = []
    for i, p in enumerate(preds):
        if p == -1:
            alerts.append(f"Row {i}: ⚠️ Threat Detected")

    return alerts