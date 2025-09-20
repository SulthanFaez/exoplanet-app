import streamlit as st
import os
import pandas as pd
import numpy as np
import joblib

# -------------------------------
# 1. Load Trained Model
# -------------------------------
@st.cache_resource
def load_model():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(BASE_DIR, "xgbexoplanet.pkl")
    # Replace with your model file path
    return joblib.load(model_path)


model = load_model()

# -------------------------------
# 2. Model Expected Features
# -------------------------------
expected_features = [
    'koi_period', 'koi_time0bk', 'koi_impact',
    'koi_duration', 'koi_depth', 'koi_prad', 'koi_teq', 'koi_insol',
    'koi_model_snr', 'koi_tce_plnt_num', 'koi_steff', 'koi_slogg',
    'koi_srad', 'ra', 'dec', 'koi_kepmag'
]

# -------------------------------
# 3. Label Mapping
# -------------------------------
label_map = {0: "FALSE POSITIVE", 1: "CONFIRMED"}

# -------------------------------
# 4. App Title
# -------------------------------
st.title("NASA Exoplanet Classifier")
st.write("Predict whether a planet candidate is CONFIRMED or FALSE POSITIVE.")

# -------------------------------
# 5. Manual Input for All Features
# -------------------------------
st.header("Enter All Planetary Features Manually")

user_input = {}

# Organize features in expandable sections
# with st.expander("Flags"):
#     user_input['koi_score'] = st.number_input("koi_score", value=0.0, step=0.1)
#     user_input['koi_fpflag_nt'] = st.number_input("koi_fpflag_nt", value=0, step=1)
#     user_input['koi_fpflag_ss'] = st.number_input("koi_fpflag_ss", value=0, step=1)
#     user_input['koi_fpflag_co'] = st.number_input("koi_fpflag_co", value=0, step=1)
#     user_input['koi_fpflag_ec'] = st.number_input("koi_fpflag_ec", value=0, step=1)

with st.expander("Orbital Parameters"):
    user_input['koi_period'] = st.number_input("koi_period (days)", value=0.0, step=0.1)
    user_input['koi_time0bk'] = st.number_input("koi_time0bk", value=0.0, step=0.1)
    user_input['koi_impact'] = st.number_input("koi_impact", value=0.0, step=0.01)
    user_input['koi_duration'] = st.number_input("koi_duration (hours)", value=0.0, step=0.01)
    user_input['koi_depth'] = st.number_input("koi_depth (ppm)", value=0.0, step=0.01)
    user_input['koi_prad'] = st.number_input("koi_prad (Earth radii)", value=0.0, step=0.01)
    user_input['koi_teq'] = st.number_input("koi_teq (K)", value=0.0, step=0.1)
    user_input['koi_insol'] = st.number_input("koi_insol", value=0.0, step=0.01)
    user_input['koi_model_snr'] = st.number_input("koi_model_snr", value=0.0, step=0.01)
    user_input['koi_tce_plnt_num'] = st.number_input("koi_tce_plnt_num", value=0, step=1)

with st.expander("Stellar Parameters"):
    user_input['koi_steff'] = st.number_input("koi_steff (K)", value=0.0, step=1.0)
    user_input['koi_slogg'] = st.number_input("koi_slogg", value=0.0, step=0.01)
    user_input['koi_srad'] = st.number_input("koi_srad (Solar radii)", value=0.0, step=0.01)
    user_input['ra'] = st.number_input("ra", value=0.0, step=0.01)
    user_input['dec'] = st.number_input("dec", value=0.0, step=0.01)
    user_input['koi_kepmag'] = st.number_input("koi_kepmag", value=0.0, step=0.01)

manual_data = pd.DataFrame([user_input])

# -------------------------------
# 6. CSV Upload Option
# -------------------------------
st.header("Or Upload CSV File")
uploaded_file = st.file_uploader("Upload CSV with planetary features", type=["csv"])
if uploaded_file is not None:
    csv_data = pd.read_csv(uploaded_file)
    st.write("Preview of uploaded CSV:")
    st.dataframe(csv_data.head())
else:
    csv_data = None

# -------------------------------
# 7. Prepare Data Function
# -------------------------------
def prepare_data(df):
    df = df.copy()
    for col in expected_features:
        if col not in df.columns:
            df[col] = 0
    df = df[expected_features]
    return df

# -------------------------------
# 8. Make Predictions
# -------------------------------
if st.button("Predict"):
    if csv_data is not None:
        input_data = prepare_data(csv_data)
    else:
        input_data = prepare_data(manual_data)

    preds = model.predict(input_data)
    pred_proba = model.predict_proba(input_data)

    # Map numerical predictions to human-readable labels
    pred_labels = [label_map[p] for p in preds]
    result_df = pd.DataFrame(pred_labels, columns=["Prediction"])
    prob_df = pd.DataFrame(np.round(pred_proba, 3), columns=[label_map[c] for c in model.classes_])

    st.header("Predictions")
    st.dataframe(pd.concat([result_df, prob_df], axis=1))

    st.success("Prediction complete!")