import streamlit as st
import pandas as pd
import joblib

from pathlib import Path

st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="👥",
    layout="wide"
)

MODEL_PATH = Path("models/kmeans_model.pkl")
SCALER_PATH = Path("models/scaler.pkl")

SEGMENT_NAMES = {
    0: "Average Income - Average Spending",
    1: "High Income - High Spending",
    2: "Low Income - High Spending",
    3: "High Income - Low Spending",
    4: "Low Income - Low Spending",
}

@st.cache_resource
def load_models():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

st.title("👥 Customer Segmentation App")
st.write(
    "Use K-Means clustering to identify customer segments "
    "based on annual income and spending score."
)

st.info(
    "This educational tool groups customers by income and spending "
    "patterns. Segment labels are descriptive, not predictions of "
    "individual customer behavior."
)

model, scaler = load_models()

st.subheader("Enter Customer Information")

col1, col2 = st.columns(2)

with col1:
    income = st.number_input(
        "Annual Income (thousands of dollars)",
        min_value=0.0,
        max_value=200.0,
        value=50.0,
        step=1.0
    )

with col2:
    spending = st.slider(
        "Spending Score",
        min_value=1,
        max_value=100,
        value=50
    )

if st.button("Identify Customer Segment", type="primary"):
    input_data = pd.DataFrame({
        "Annual Income (k$)": [income],
        "Spending Score (1-100)": [spending]
    })

    scaled_input = scaler.transform(input_data)
    cluster = int(model.predict(scaled_input)[0])
    segment = SEGMENT_NAMES.get(cluster, f"Cluster {cluster}")

    st.success(f"Customer Segment: {segment}")
    st.metric("Cluster ID", cluster)

st.divider()

st.subheader("About This Project")
st.markdown(
    """
    - **Algorithm:** K-Means Clustering
    - **Dataset:** Mall Customers (200 records)
    - **Features:** Annual Income and Spending Score
    - **Evaluation:** Silhouette Score
    - **Libraries:** Python, Pandas, Scikit-learn, Streamlit
    """
)
