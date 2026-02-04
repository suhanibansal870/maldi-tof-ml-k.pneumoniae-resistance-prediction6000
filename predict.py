import streamlit as st
import json
import pandas as pd
import pickle
import numpy as np
from datetime import datetime
from statistics import mode

from constants import ALL_COLUMNS

# constants
JSONS = ["Sample_With6000_1.json", "Sample_With6000_2.json", "Sample_With6000_3.json", "Sample_With6000_4.json", "Sample_With6000_5.json"]

IMAGE_ADDRESS = "https://biolabtests.com/wp-content/uploads/Microbial-Top-Facts-Klebsiella-pneumoniae.png"

# Add an image
st.image(IMAGE_ADDRESS, caption="Classification")

st.set_page_config(
    page_title="K. pneumoniae • Ertapenem S/R Predictor",
    page_icon="🧬",
    layout="wide"
)

@st.cache_resource
def load_pickle_model(model_path: str):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model

# load all three models
logistic_regression_model = load_pickle_model("LR")
mlp_model = load_pickle_model("MLP")
rf_model = load_pickle_model("RandomForest")

MODEL_METRICS = {
    "Logistic Regression": {
        "precision": 0.85,
        "recall": 0.82,
        "f1_score": 0.83,
        "accuracy": 0.88,
    },
    "Multi-Layer Perceptron": {
        "precision": 0.87,
        "recall": 0.84,
        "f1_score": 0.85,
        "accuracy": 0.89,
    },
    "K-Nearest Neighbors": {
        "precision": 0.83,
        "recall": 0.80,
        "f1_score": 0.81,
        "accuracy": 0.86,
    },
}

st.title("Klebsiella pneumoniae – Ertapenem Susceptibility")
st.subheader("Predict Susceptible (S) or Resistant (R) from JSON features")

# Upload section
st.header("📤 Upload JSON Data")
uploaded_file = st.file_uploader(
    "Upload your spectral data (JSON only)",
    type=["json"],
    accept_multiple_files=False,
    help="Upload a JSON file containing spectral data",
)

# Sidebar example JSON
with st.sidebar:
    st.subheader("Download Example Json")
    json_name = st.selectbox("Select Example Json", JSONS)

    with open(json_name, "r") as f:
        json_data = json.load(f)

    example_df = pd.json_normalize(json_data)

    missing_columns = [col for col in ALL_COLUMNS if col not in example_df.columns]
    if missing_columns:
        st.error("Example JSON is missing required features.")
        st.stop()

    example_df = example_df[ALL_COLUMNS]
    filtered_example_json = example_df.to_dict(orient="records")[0]

    with st.expander("Example Json"):
        st.json(filtered_example_json)

    st.download_button(
        label="Download Example Json",
        data=json.dumps(filtered_example_json, indent=4),
        file_name=json_name,
        mime="application/json",
    )

# Helper functions
def get_prediction_label(prediction_value):
    if prediction_value == 0:
        return "Resistant"
    elif prediction_value == 1:
        return "Susceptible (S)"
    return f"Unknown ({prediction_value})"

def get_short_prediction_label(prediction_value):
    return "S" if prediction_value == 1 else "Resistant"

def get_prediction_confidence(model, df, prediction_value):
    try:
        if hasattr(model, "predict_proba"):
            return model.predict_proba(df)[0][prediction_value] * 100
    except Exception:
        pass
    return None

#file processing 
if uploaded_file is not None:
    try:
        json_data = json.load(uploaded_file)
        sample_id = uploaded_file.name.replace(".json", "")

        st.success(f"✅ File '{sample_id}' successfully uploaded and processed!")

        with st.expander("📊 View Uploaded Data"):
            st.json(json_data)

        # df validation
        try:
            df = pd.json_normalize(json_data)

            missing_columns = [col for col in ALL_COLUMNS if col not in df.columns]
            if missing_columns:
                st.error(
                    "❌ The uploaded JSON file does not contain all required columns.",
                    icon="⚠️",
                )
                with st.expander("Missing Columns"):
                    st.write(missing_columns)
                st.stop()

            df = df[ALL_COLUMNS]

        except Exception:
            st.error("❌ Failed to process uploaded JSON.", icon="⚠️")
            st.stop()

        # ---- Prediction ----
        if st.button("RUN PREDICTION", type="primary"):
            with st.spinner("Analyzing Spectral data with all models..."):
                lr_prediction = logistic_regression_model.predict(df)[0]
                mlp_prediction = mlp_model.predict(df)[0]
                rf_prediction = rf_model.predict(df)[0]

                results_data = [
                    {
                        "Sample ID": sample_id,
                        "Model": "Logical Regression",
                        "Precision": MODEL_METRICS["Logistic Regression"]["precision"],
                        "Recall": MODEL_METRICS["Logistic Regression"]["recall"],
                        "F1-Score": MODEL_METRICS["Logistic Regression"]["f1_score"],
                        "Accuracy": MODEL_METRICS["Logistic Regression"]["accuracy"],
                        "Final Outcome": get_short_prediction_label(lr_prediction),
                    },
                    {
                        "Sample ID": sample_id,
                        "Model": "MLP",
                        "Precision": MODEL_METRICS["Multi-Layer Perceptron"]["precision"],
                        "Recall": MODEL_METRICS["Multi-Layer Perceptron"]["recall"],
                        "F1-Score": MODEL_METRICS["Multi-Layer Perceptron"]["f1_score"],
                        "Accuracy": MODEL_METRICS["Multi-Layer Perceptron"]["accuracy"],
                        "Final Outcome": get_short_prediction_label(mlp_prediction),
                    },
                    {
                        "Sample ID": sample_id,
                        "Model": "RF",
                        "Precision": MODEL_METRICS["K-Nearest Neighbors"]["precision"],
                        "Recall": MODEL_METRICS["K-Nearest Neighbors"]["recall"],
                        "F1-Score": MODEL_METRICS["K-Nearest Neighbors"]["f1_score"],
                        "Accuracy": MODEL_METRICS["K-Nearest Neighbors"]["accuracy"],
                        "Final Outcome": get_short_prediction_label(rf_prediction),
                    },
                ]

                results_df = pd.DataFrame(results_data)
                st.dataframe(results_df, use_container_width=True, hide_index=True)

                st.divider()
                st.subheader("🎯 Consensus Analysis")

                final_prediction_value = mode(
                    [lr_prediction, mlp_prediction, rf_prediction]
                )
                final_prediction = get_prediction_label(final_prediction_value)

                st.success(final_prediction)

    except json.JSONDecodeError:
        st.error("❌ Invalid JSON file. Please upload a valid JSON file.", icon="⚠️")
    except Exception as error:
        st.error(f"An error occurred: {error}", icon="❌")
