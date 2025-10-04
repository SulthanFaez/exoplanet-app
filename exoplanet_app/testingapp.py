import streamlit as st
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import pickle
import io

@st.cache_resource
# def load_default_model():
#     BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#     model_path = os.path.join(BASE_DIR, "tabpfn_exoplanet.pkl")  # your trained model
#     return joblib.load(model_path)

def load_default_model():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(BASE_DIR, "tabpfn_exoplanet.pth")
    model = torch.load(model_path, map_location="cpu", weights_only=False)
    model.devices_ = [torch.device("cpu")]        # devices list
    model.use_cuda = False
    return model

def load_model_file(uploaded_file):
    return joblib.load(uploaded_file)

model = load_default_model()

expected_features = [
    'koi_period', 'koi_time0bk', 'koi_impact',
    'koi_duration', 'koi_depth', 'koi_prad', 'koi_teq', 'koi_insol',
    'koi_model_snr', 'koi_tce_plnt_num', 'koi_steff', 'koi_slogg',
    'koi_srad', 'ra', 'dec', 'koi_kepmag'
]

label_map = {0: "FALSE POSITIVE", 1: "CONFIRMED"}

def prepare_data(df):
    df = df.copy()
    for col in expected_features:
        if col not in df.columns:
            df[col] = 0
    df = df[expected_features]
    return df

def plot_confusion_matrix(cm, class_names):
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names,
                ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    st.pyplot(fig)


st.title("🔭 Trigospace Exoplanet Classifier")

tab1, tab2, tab3 = st.tabs(["Predict", "Train / Retrain Model", "Load & Predict"])

with tab1:
    st.header("Predict Exoplanet Candidate")

    st.subheader("Enter All Features Manually")
    user_input = {}

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

    st.subheader("Or Upload CSV File")
    uploaded_file = st.file_uploader("Upload CSV with planetary features", type=["csv"])
    if uploaded_file is not None:
        csv_data = pd.read_csv(
            uploaded_file,
            sep=",",
            comment="#",   
            engine="python",
            on_bad_lines="skip"  
        )
        
        st.write("Preview of uploaded CSV:")
        st.dataframe(csv_data.head())
    else:
        csv_data = None

if st.button("Predict"):
    if csv_data is not None:
        input_data = prepare_data(csv_data)
    else:
        input_data = prepare_data(manual_data)

    X = input_data  # make sure it's a NumPy array

    chunk_size = 32
    all_preds = []
    all_probs = []

    # Loop in batches
    for start in range(0, len(X), chunk_size):
        batch = X[start:start + chunk_size]
        batch_preds = model.predict(batch)
        batch_probs = model.predict_proba(batch)
        all_preds.extend(batch_preds)
        all_probs.extend(batch_probs)

    preds = np.array(all_preds)
    pred_proba = np.array(all_probs)

    # Map predictions to labels
    pred_labels = [label_map[p] for p in preds]

    # Create DataFrames
    result_df = pd.DataFrame(pred_labels, columns=["Prediction"])
    prob_df = pd.DataFrame(np.round(pred_proba, 3), columns=[label_map[c] for c in model.classes_])

    # Display in Streamlit
    st.header("Predictions")
    st.dataframe(pd.concat([result_df, prob_df], axis=1))

    st.success("Prediction complete!")


with tab2:
    st.header("Train or Retrain Model")
    st.write("Upload a dataset including the **target** column to retrain the model.")

    train_file = st.file_uploader("Upload training CSV", type=['csv'])
    target_column = st.text_input("Target column name", "target")

    model_choice = st.selectbox(
        "Pick Model to Train",
        ["Random Forest", "XGBoost", "Logistic Regression", "TabPFN"]
    )

    if train_file is not None:
        train_data = pd.read_csv(train_file, sep=",", comment='#', engine="python", on_bad_lines="skip")
        st.write("Preview of training data:")
        st.write(train_data.head())

        numeric_cols = train_data.select_dtypes(include=[np.number]).columns.tolist()
        default_continuous = [c for c in numeric_cols if c != target_column]
        continuous_columns = st.multiselect(
            "Select continuous (numerical) columns:",
            options=numeric_cols,
            default=default_continuous
        )

        st.write("Selected continuous columns:")
        st.write(continuous_columns)    

    if model_choice == "Random Forest":
        from sklearn.ensemble import RandomForestClassifier
        n_estimators = st.slider("Number of Trees (n_estimators)", 10, 500, 100, step=10)
        max_depth = st.slider("Max Depth", 1, 50, 10, step=1)
    elif model_choice == "XGBoost":
        import xgboost as xgb
        n_estimators = st.slider("Number of Boosting Rounds", 10, 1000, 100, step=10)
        lr_str = st.text_input("Learning Rate", "0.001")
        try:
            learning_rate = float(lr_str)
        except ValueError:
            st.error("Please enter a valid number.")
            learning_rate = 0.001

        max_depth = st.slider("Max Depth", 1, 50, 6, step=1)
    elif model_choice == "Logistic Regression":
        from sklearn.linear_model import LogisticRegression
        C = st.number_input("Inverse Regularization Strength (C)", 0.01, 10.0, 1.0)
    elif model_choice == "TabPFN":
        from tabpfn import TabPFNClassifier


    if st.button("Train Model"):
        if train_file is not None:
            X = train_data.drop(target_column, axis=1)
            y = train_data[target_column]

            if y.dtype == 'object' or y.dtype.name == 'category':
                st.warning(f"Target column '{target_column}' contains string labels. Encoding to numbers automatically.")
                label_map_train = {label: idx for idx, label in enumerate(y.unique())}
                y_encoded = y.map(label_map_train)
            else:
                y_encoded = y

            X = X[continuous_columns]
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

            with st.spinner("Training..."):
                if model_choice == "Random Forest":
                    clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
                elif model_choice == "XGBoost":
                    clf = xgb.XGBClassifier(
                        n_estimators=n_estimators,
                        learning_rate=learning_rate,
                        max_depth=max_depth,
                        random_state=42,
                        use_label_encoder=False,
                        eval_metric='logloss'
                    )
                elif model_choice == "Logistic Regression":
                    clf = LogisticRegression(C=C, max_iter=500)
                elif model_choice == "TabPFN":
                    clf = TabPFNClassifier

                clf.fit(X_train, y_train)
                joblib.dump(clf, f"{model_choice}_exoplanet_model.pkl")

                with open(f"{model_choice}_exoplanet_model.pkl", "rb") as f:
                    st.download_button("Download Trained Model", f, file_name="xgbexoplanet.pkl")

            st.success(f"{model_choice} model retrained and saved!")

            y_pred = clf.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            st.metric("Accuracy", f"{acc:.2%}")

            cm = confusion_matrix(y_test, y_pred)
            st.write("Confusion Matrix:")
            plot_confusion_matrix(cm, np.unique(y))
        else:
            st.error("Please upload a training CSV first.")

with tab3:
    st.header("Load Pretrained Model and Predict")

    model_file = st.file_uploader("Upload your trained model (.pkl)", type=["pkl"])
    data_file = st.file_uploader("Upload CSV for Prediction", type=["csv"])

    if data_file is not None:
        df_pred = pd.read_csv(data_file, sep=",", comment='#', engine="python", on_bad_lines="skip")
        st.write("Preview of uploaded data:")
        st.dataframe(df_pred.head())

        numeric_cols = df_pred.select_dtypes(include=[np.number]).columns.tolist()
        selected_columns = st.multiselect(
            "Select features to use for prediction:",
            options=numeric_cols,
            default=numeric_cols
        )
        df_pred = df_pred[selected_columns]

    if model_file is not None and data_file is not None:
        if st.button("Run Prediction"):
            try:
                if model_file.name.endswith(".pkl"):
                    import io
                    import joblib
                    model_bytes = model_file.read()
                    model = joblib.load(io.BytesIO(model_bytes))

                    preds = model.predict(df_pred)
                    if hasattr(model, "predict_proba"):
                        probs = model.predict_proba(df_pred)
                        prob_df = pd.DataFrame(probs, columns=[f"Class {c}" for c in model.classes_])
                        st.dataframe(pd.concat([pd.Series(preds, name="Prediction"), prob_df], axis=1))
                    else:
                        st.write("Predictions:", preds)

            except Exception as e:
                st.error(f"Error while loading model or predicting: {e}")






