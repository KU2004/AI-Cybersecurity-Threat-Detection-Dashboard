from src.data_preprocessing import load_data, preprocess_data
from src.feature_engineering import select_features
from src.model import train_model, evaluate_model
from src.detect import detect_threat
from src.visualize import plot_confusion_matrix

print("🚀 Starting AI Cybersecurity Threat Detection System...")

# Load Data
df = load_data("data/KDDTrain+.txt")

print("\n✅ Dataset Loaded:")
print(df.head())

# Preprocess
df = preprocess_data(df)

print("\n✅ After Preprocessing:")
print(df.head())

# Features
X, y = select_features(df)

# Train
model = train_model(X)

# Evaluate
preds = evaluate_model(model, X, y)

# Plot (THIS WILL OPEN WINDOW)
plot_confusion_matrix(y, preds)

# Detect Threats
alerts = detect_threat(X[:50])

print("\n🚨 ALERTS:")
for alert in alerts:
    print(alert)