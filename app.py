import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, classification_report

from src.data_preprocessing import load_data, preprocess_data
from src.feature_engineering import select_features
from src.model import train_model
from src.detect import detect_threat

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Cyber Threat Detection",
    page_icon="🛡️",
    layout="wide"
)

# ================= CUSTOM CSS (🔥 COOL UI) =================
st.markdown("""
<style>
/* Background */
.stApp {
    background: linear-gradient(135deg, #020617, #0f172a);
    color: white;
}

/* Titles */
h1, h2, h3 {
    color: #38bdf8;
}

/* Cards */
.metric-card {
    background: rgba(15, 23, 42, 0.8);
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    box-shadow: 0 0 15px rgba(56,189,248,0.3);
}

/* Alerts */
.alert-box {
    background: #7f1d1d;
    padding: 10px;
    border-radius: 10px;
    margin-bottom: 8px;
    color: white;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #020617;
}

/* Button */
.stButton>button {
    background: linear-gradient(90deg, #38bdf8, #6366f1);
    color: white;
    border-radius: 10px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ================= SIDEBAR =================
st.sidebar.title("🛡️ Cyber Panel")
page = st.sidebar.radio("Navigate", [
    "🏠 Dashboard",
    "📂 Data",
    "⚙️ Processing",
    "🤖 Model",
    "🚨 Detection"
])

# ================= LOAD DATA =================
df = load_data("data/KDDTrain+.txt")

# ================= HEADER =================
st.title("🛡️ AI Cybersecurity Threat Detection System")
st.markdown("### 🔍 Real-Time Network Intrusion Detection Dashboard")
st.markdown("---")

# ================= DASHBOARD =================
if page == "🏠 Dashboard":
    st.subheader("📊 System Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.markdown(f"<div class='metric-card'><h3>{len(df)}</h3><p>Total Records</p></div>", unsafe_allow_html=True)
    col2.markdown(f"<div class='metric-card'><h3>{df.shape[1]}</h3><p>Features</p></div>", unsafe_allow_html=True)
    col3.markdown(f"<div class='metric-card'><h3>{df['label'].nunique()}</h3><p>Attack Types</p></div>", unsafe_allow_html=True)
    col4.markdown(f"<div class='metric-card'><h3>ACTIVE</h3><p>Status</p></div>", unsafe_allow_html=True)

    st.markdown("---")

    st.subheader("📡 Live Traffic Sample")
    st.dataframe(df.sample(10))

# ================= DATA =================
elif page == "📂 Data":
    st.subheader("📂 Raw Dataset")
    st.dataframe(df)

# ================= PROCESSING =================
elif page == "⚙️ Processing":
    st.subheader("⚙️ Data Processing Pipeline")

    df_clean = preprocess_data(df)

    st.success("✔ Data Cleaned & Encoded")

    st.markdown("### Preview")
    st.dataframe(df_clean.head())

# ================= MODEL =================
elif page == "🤖 Model":
    st.subheader("🤖 Model Training & Evaluation")

    df_clean = preprocess_data(df)
    X, y = select_features(df_clean)

    if st.button("🚀 Train Model"):
        model = train_model(X)

        st.success("Model Trained Successfully")

        preds = model.predict(X)
        preds = [1 if p == -1 else 0 for p in preds]

        # Classification report
        st.markdown("### 📊 Performance Report")
        st.code(classification_report(y, preds))

        # Confusion matrix
        st.markdown("### 📉 Confusion Matrix")
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay.from_predictions(y, preds, ax=ax)
        st.pyplot(fig)

# ================= DETECTION =================
elif page == "🚨 Detection":
    st.subheader("🚨 Threat Detection Engine")

    df_clean = preprocess_data(df)
    X, y = select_features(df_clean)

    model = train_model(X)
    alerts = detect_threat(X[:100])

    st.markdown("### ⚠️ Threat Alerts")

    if alerts:
        for alert in alerts[:20]:
            st.markdown(f"<div class='alert-box'>{alert}</div>", unsafe_allow_html=True)
    else:
        st.success("No threats detected")

    st.markdown("---")

    st.subheader("📊 Threat Distribution")

    threat_count = len(alerts)
    normal_count = 100 - threat_count

    fig, ax = plt.subplots()
    ax.pie(
        [threat_count, normal_count],
        labels=["Threats", "Normal"],
        autopct="%1.1f%%"
    )
    st.pyplot(fig)

    st.markdown("---")

    st.subheader("💻 System Log (Terminal View)")
    for alert in alerts[:10]:
        st.code(alert)