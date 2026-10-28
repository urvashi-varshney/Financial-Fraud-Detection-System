import hashlib
import io
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Fraud Detection",
    page_icon="💳",
    layout="wide",
)

# ---------------- CUSTOM CSS ----------------
st.markdown(
    """
    <style>
    .stApp {
        background-color: #0b1120;
        color: #f1f5f9;
    }
    [data-testid="stSidebar"] {
        background-color: #111827;
    }
    div[data-testid="stMetric"] {
        background-color: #172033;
        border: 1px solid #29364d;
        padding: 18px;
        border-radius: 12px;
    }
    div[data-testid="stMetricValue"] {
        color: #38bdf8;
    }
    h1, h2, h3 {
        color: #f8fafc;
    }
    .stButton button {
        border-radius: 8px;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("💳 AI-Based Financial Fraud Detection")
st.caption(
    "Machine-learning dashboard powered by Logistic Regression "
    "and Random Forest."
)

# ---------------- SIDEBAR ----------------
st.sidebar.title("🛡️ Fraud Detection")
page = st.sidebar.radio(
    "Select Page",
    ["Dashboard", "Dataset Explorer", "Fraud Prediction", "Model Evaluation"],
)
st.sidebar.divider()
st.sidebar.info(
    "Educational demonstration only. Do not use these predictions "
    "to make real financial decisions."
)

uploaded_file = st.sidebar.file_uploader(
    "Upload credit-card CSV",
    type=["csv"],
)


# ---------------- LOAD AND VALIDATE DATA ----------------
@st.cache_data(show_spinner=False)
def load_uploaded_csv(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(file_bytes))


def load_local_csv() -> pd.DataFrame | None:
    for filename in ("creditcard.csv", "creditcard.csv.csv"):
        if os.path.isfile(filename):
            return pd.read_csv(filename)
    return None


def validate_data(df: pd.DataFrame):
    if df.empty:
        raise ValueError("The dataset is empty.")
    if not df.columns.is_unique:
        raise ValueError("Duplicate column names were found.")
    if "Class" not in df.columns:
        raise ValueError("The dataset must contain a 'Class' column.")
    if df.isnull().values.any():
        raise ValueError("Missing values were found. Clean the CSV and try again.")

    X = df.drop(columns=["Class"])
    y = df["Class"]

    if X.shape[1] == 0:
        raise ValueError("No feature columns were found.")
    if not all(pd.api.types.is_numeric_dtype(X[col]) for col in X.columns):
        raise ValueError("Every feature column must contain numeric values.")
    if not pd.api.types.is_numeric_dtype(y):
        raise ValueError("The Class column must contain numeric labels.")

    try:
        finite = np.isfinite(X.to_numpy(dtype=np.float64)).all()
    except (TypeError, ValueError):
        finite = False
    if not finite:
        raise ValueError("Features contain infinite or invalid values.")

    if not set(pd.unique(y)).issubset({0, 1}) or set(pd.unique(y)) != {0, 1}:
        raise ValueError("Class must contain both labels: 0 (legitimate) and 1 (fraud).")
    if y.value_counts().min() < 2:
        raise ValueError("At least two transactions are required in each class.")

    return X, y.astype(int)


@st.cache_resource(show_spinner="Training both machine-learning models...")
def train_models(data_fingerprint: str, csv_text: str):
    # Use a CSV string as a stable cache input, rather than a mutable DataFrame.
    df = pd.read_csv(io.StringIO(csv_text))
    X = df.drop(columns=["Class"])
    y = df["Class"].astype(int)

    # Keep the test split stratified so the rare fraud class is represented.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models_to_train = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
    }

    trained_models = {}
    metrics = {}

    for name, model in models_to_train.items():
        model.fit(X_train_scaled, y_train)
        predictions = model.predict(X_test_scaled)
        cm = confusion_matrix(y_test, predictions, labels=[0, 1])

        metrics[name] = {
            "Accuracy": accuracy_score(y_test, predictions),
            "Precision": precision_score(y_test, predictions, zero_division=0),
            "Recall": recall_score(y_test, predictions, zero_division=0),
            "F1 Score": f1_score(y_test, predictions, zero_division=0),
            "Confusion Matrix": cm,
            "Report": classification_report(
                y_test,
                predictions,
                labels=[0, 1],
                target_names=["Legitimate", "Fraud"],
                zero_division=0,
                output_dict=True,
            ),
        }
        trained_models[name] = model

    # Recall is prioritized because missing fraudulent transactions is costly.
    best_name = max(
        trained_models,
        key=lambda name: (
            metrics[name]["Recall"],
            metrics[name]["Precision"],
        ),
    )

    # Store the held-out test features and their original CSV row indices.
    test_rows = X_test.copy()
    test_rows["_original_row"] = X_test.index.to_numpy()

    return (
        trained_models,
        metrics,
        scaler,
        best_name,
        test_rows,
        y_test,
    )


# ---------------- LOAD DATASET ----------------
try:
    if uploaded_file is not None:
        df = load_uploaded_csv(uploaded_file.getvalue())
    else:
        df = load_local_csv()
except (pd.errors.ParserError, UnicodeDecodeError, OSError, ValueError) as exc:
    st.error(f"Could not read the CSV file: {exc}")
    st.stop()

if df is None:
    st.warning(
        "Upload your credit-card CSV using the sidebar, or place "
        "`creditcard.csv` in the same folder as `app.py`."
    )
    st.code("your-project/\n├── app.py\n└── creditcard.csv")
    st.stop()

try:
    X, y = validate_data(df)
    # A stable fingerprint ensures a changed dataset triggers retraining.
    csv_text = df.to_csv(index=False)
    fingerprint = hashlib.sha256(csv_text.encode("utf-8")).hexdigest()
    (
        models,
        metrics,
        scaler,
        best_model_name,
        test_rows,
        y_test,
    ) = train_models(fingerprint, csv_text)
except Exception as exc:
    st.error(f"Dataset validation or model training failed: {exc}")
    st.stop()


# ---------------- DASHBOARD ----------------
if page == "Dashboard":
    st.subheader("📊 Dashboard Overview")

    total = len(df)
    fraud_count = int((y == 1).sum())
    legitimate_count = int((y == 0).sum())
    fraud_rate = fraud_count / total * 100 if total else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Transactions", f"{total:,}")
    col2.metric("Fraud Transactions", f"{fraud_count:,}")
    col3.metric("Legitimate Transactions", f"{legitimate_count:,}")
    col4.metric("Fraud Rate", f"{fraud_rate:.3f}%")

    st.divider()
    left, right = st.columns(2)

    with left:
        st.subheader("Transaction Distribution")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(
            ["Legitimate", "Fraud"],
            [legitimate_count, fraud_count],
        )
        ax.set_ylabel("Number of transactions")
        ax.set_title("Class Distribution")
        st.pyplot(fig)
        plt.close(fig)

    with right:
        st.subheader("Best Model Performance")
        selected_metrics = metrics[best_model_name]
        st.success(f"Selected model: {best_model_name}")
        st.metric("Accuracy", f"{selected_metrics['Accuracy']:.2%}")
        st.metric("Fraud Precision", f"{selected_metrics['Precision']:.2%}")
        st.metric("Fraud Recall", f"{selected_metrics['Recall']:.2%}")
        st.metric("Fraud F1 Score", f"{selected_metrics['F1 Score']:.2%}")

    st.info(
        "Model selection prioritizes fraud recall, then precision. "
        "A high recall catches more fraud but may also flag legitimate transactions."
    )


# ---------------- DATASET EXPLORER ----------------
elif page == "Dataset Explorer":
    st.subheader("📁 Dataset Explorer")
    st.write("Dataset shape:", df.shape)
    st.write("Number of features:", X.shape[1])

    st.subheader("Find Legitimate and Fraud Transactions")
    transaction_filter = st.selectbox(
        "Choose which transactions to view",
        ["All Transactions", "Legitimate Only", "Fraud Only"],
    )
    if transaction_filter == "Legitimate Only":
        filtered_df = df.loc[df["Class"] == 0].copy()
    elif transaction_filter == "Fraud Only":
        filtered_df = df.loc[df["Class"] == 1].copy()
    else:
        filtered_df = df.copy()

    # CSV line numbers start at 1 for the header, so the first data row is 2.
    # The internal DataFrame index remains unchanged for model operations.
    filtered_df.insert(
        0,
        "CSV Row Number",
        filtered_df.index.to_series().astype(int) + 2,
    )
    filtered_df.insert(
        1,
        "Transaction Type",
        filtered_df["Class"].map({0: "Legitimate", 1: "Fraud"}),
    )
    st.write(f"Showing **{len(filtered_df):,}** transaction(s).")
    st.dataframe(filtered_df, use_container_width=True, hide_index=True)

    st.download_button(
        "Download Filtered Transactions",
        data=filtered_df.to_csv(index=False).encode("utf-8"),
        file_name="filtered_transactions.csv",
        mime="text/csv",
    )

    st.subheader("Class Distribution")
    distribution = pd.DataFrame(
        {
            "Transaction Type": ["Legitimate", "Fraud"],
            "Count": [int((y == 0).sum()), int((y == 1).sum())],
        }
    )
    st.dataframe(distribution, use_container_width=True, hide_index=True)

    st.subheader("Column Information")
    column_info = pd.DataFrame(
        {
            "Column": df.columns,
            "Data Type": df.dtypes.astype(str).to_numpy(),
            "Missing Values": df.isnull().sum().to_numpy(),
        }
    )
    st.dataframe(column_info, use_container_width=True, hide_index=True)

    st.download_button(
        "Download Dataset CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="validated_transactions.csv",
        mime="text/csv",
    )


# ---------------- FRAUD PREDICTION ----------------
elif page == "Fraud Prediction":
    st.subheader("🔍 Transaction Fraud Prediction")
    st.write(
        "Choose a transaction from the held-out test set. "
        "These rows were not used to train the models."
    )

    test_rows = test_rows.reset_index(drop=True)
    row_options = test_rows["_original_row"].astype(int).tolist()

    original_row = st.selectbox(
        "Select original CSV row number",
        options=row_options,
        format_func=lambda row_index: str(row_index + 2),
    )
    model_choice = st.selectbox(
        "Choose machine-learning model",
        options=list(models.keys()),
        index=list(models.keys()).index(best_model_name),
    )

    if st.button("Analyze Transaction", type="primary"):
        selected = test_rows.loc[test_rows["_original_row"] == original_row]
        features = selected.drop(columns=["_original_row"])
        features_scaled = scaler.transform(features)

        model = models[model_choice]
        prediction = int(model.predict(features_scaled)[0])
        actual = int(df.iloc[int(original_row)]["Class"])

        if prediction == 1:
            st.error("🚨 FRAUD DETECTED")
        else:
            st.success("✅ PREDICTED AS LEGITIMATE")

        p1, p2, p3 = st.columns(3)
        p1.metric("CSV Row Number", str(original_row + 2))
        p2.metric("Actual Class", "Fraud" if actual == 1 else "Legitimate")
        p3.metric("Model Prediction", "Fraud" if prediction == 1 else "Legitimate")

        if hasattr(model, "predict_proba"):
            fraud_probability = float(model.predict_proba(features_scaled)[0][1])
            st.write("Estimated fraud probability")
            st.progress(max(0.0, min(1.0, fraud_probability)))
            st.write(f"{fraud_probability:.2%}")

        st.caption(
            "Probability is a model estimate, not proof that a transaction is fraudulent."
        )
        with st.expander("View transaction features"):
            st.dataframe(features, use_container_width=True)


# ---------------- MODEL EVALUATION ----------------
elif page == "Model Evaluation":
    st.subheader("📈 Model Comparison")

    comparison = pd.DataFrame(metrics).T[
        ["Accuracy", "Precision", "Recall", "F1 Score"]
    ]
    st.dataframe(comparison.style.format("{:.4f}"), use_container_width=True)
    st.bar_chart(comparison)

    selected_model = st.selectbox(
        "View detailed evaluation",
        options=list(models.keys()),
        index=list(models.keys()).index(best_model_name),
    )
    selected_metrics = metrics[selected_model]
    st.subheader(selected_model)

    cm = selected_metrics["Confusion Matrix"]
    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(cm, interpolation="nearest")
    ax.set(
        title="Confusion Matrix",
        xlabel="Predicted label",
        ylabel="Actual label",
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["Legitimate", "Fraud"],
        yticklabels=["Legitimate", "Fraud"],
    )
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.colorbar(image, ax=ax)
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Classification Report")
    report_df = pd.DataFrame(selected_metrics["Report"]).T
    st.dataframe(report_df.style.format("{:.3f}"), use_container_width=True)


st.divider()
st.caption("AI Financial Fraud Detection System | Educational project")
