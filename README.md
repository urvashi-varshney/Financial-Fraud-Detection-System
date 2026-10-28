# 💳 AI-Based Financial Fraud Detection System

An AI-powered financial fraud detection dashboard built using Python, Streamlit, and Machine Learning. The system analyzes credit card transaction data and predicts whether a transaction is **Legitimate** or **Fraudulent** using Logistic Regression and Random Forest classifiers.

## 📌 Project Overview

Financial fraud is a major challenge in digital payments and online banking. This project demonstrates how machine-learning techniques can help identify potentially fraudulent credit card transactions by analyzing transaction features and comparing model predictions.

The application provides an interactive dashboard for exploring transaction data, predicting fraud on held-out test transactions, and evaluating the performance of machine-learning models.

> **Note:** This project is intended for educational and demonstration purposes only. Predictions should not be used as the sole basis for real financial decisions.

## ✨ Features

- **Dashboard Overview:** Displays total transactions, fraud transactions, legitimate transactions, fraud rate, and model performance.
- **Dataset Explorer:** Explore transaction records and filter legitimate or fraudulent transactions.
- **Fraud Prediction:** Select a held-out test transaction and choose a machine-learning model to analyze it.
- **Fraud Probability:** Displays the model-estimated probability of fraud when supported by the selected classifier.
- **Model Evaluation:** Compare model accuracy, precision, recall, and F1 score.
- **Confusion Matrix:** Visualizes correct and incorrect predictions for each class.
- **Classification Report:** Provides detailed model evaluation metrics.
- **CSV Upload:** Upload a credit card transaction dataset through the sidebar.
- **Data Validation:** Checks for missing values, invalid features, duplicate column names, and valid class labels.
- **Download Options:** Export filtered transactions and validated dataset records as CSV files.
- **Interactive Interface:** Uses a dark-themed dashboard with separate pages for different tasks.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | Interactive web dashboard |
| Pandas | Data loading and manipulation |
| NumPy | Numerical operations |
| Scikit-learn | Machine-learning models and evaluation |
| Matplotlib | Data visualization |

## 🤖 Machine-Learning Algorithms

### 1. Logistic Regression

A classification algorithm used to estimate the probability that a transaction belongs to the fraudulent class. The implementation uses balanced class weights to help address the imbalance between legitimate and fraudulent transactions.

### 2. Random Forest

An ensemble-learning algorithm that combines multiple decision trees to classify transactions. It also uses balanced class weights to account for the unequal distribution of transaction classes.

### Data Preprocessing and Training

- Separates the `Class` column from the input features.
- Splits the dataset into training and testing sets using an 80:20 ratio.
- Uses stratified splitting to preserve class proportions.
- Applies `StandardScaler` to scale input features.
- Evaluates both models on held-out test data.
- Selects the preferred model by prioritizing fraud recall, followed by precision.

## 📊 Dataset Information

The application expects a credit card transaction dataset in CSV format.

The dataset must contain:

- **Features:** Numeric transaction attributes, such as `Time`, `V1`, `V2`, and other available feature columns.
- **Class:** The target column identifying the transaction type.

Class labels:

- `0` — Legitimate transaction
- `1` — Fraudulent transaction

The dataset must contain both classes and must not have missing or infinite feature values.

### Dataset Statistics

The screenshots provided for this project show the following dataset distribution:

| Metric | Value |
|---|---:|
| Total Transactions | 10,000 |
| Legitimate Transactions | 9,962 |
| Fraudulent Transactions | 38 |
| Fraud Rate | 0.380% |

These figures describe the dataset shown in the screenshots. Results will change if a different dataset is uploaded.

**Dataset privacy:** The dataset file is not included in this repository by default. Use an authorized dataset and avoid uploading sensitive financial or personal information to a public repository.

## 📈 Model Evaluation

The dashboard compares the models using the following metrics:

- **Accuracy:** Proportion of all transactions classified correctly.
- **Precision:** Proportion of transactions predicted as fraud that are actually fraudulent.
- **Recall:** Proportion of actual fraudulent transactions correctly identified.
- **F1 Score:** Harmonic mean of precision and recall.

The screenshots show the following example results:

| Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| Accuracy | 99.85% | 99.90% |
| Precision | 77.78% | 100.00% |
| Recall | 87.50% | 75.00% |
| F1 Score | 82.35% | 85.71% |

These values are the results displayed in the supplied screenshots, not guaranteed results for every dataset or execution.

## 🖥️ Application Modules

### 1. Dashboard

Displays transaction totals, fraud rate, transaction distribution, and the selected model's performance metrics.

### 2. Dataset Explorer

Shows the dataset dimensions, column information, missing-value counts, class distribution, and filtered transaction records. Filtered records and validated data can be downloaded as CSV files.

### 3. Fraud Prediction

Allows users to choose a transaction from the held-out test set, select Logistic Regression or Random Forest, and view the predicted class, actual class, estimated fraud probability, and transaction features.

### 4. Model Evaluation

Displays a model comparison table, performance chart, confusion matrix, and classification report for the selected algorithm.

## 🔮 Future Enhancements

- Integrate additional machine-learning algorithms.
- Improve performance on highly imbalanced datasets.
- Add more detailed visual analytics and feature-importance analysis.
- Introduce configurable classification thresholds.
- Add secure authentication and role-based access.
- Deploy the application to a cloud hosting platform.
- Evaluate the system on larger, more diverse datasets.

## 🚀 Applications

- Educational demonstrations of financial fraud detection.
- Machine-learning and data-analysis projects.
- Credit card transaction classification experiments.
- Comparative evaluation of classification algorithms.
- Academic presentations and project demonstrations.

## ⚠️ Limitations

- Model performance depends on the quality and representativeness of the dataset.
- Imbalanced class distributions can make accuracy misleading.
- A predicted fraud probability is a model estimate, not proof of fraud.
- The application is not a production-grade financial monitoring system.
- Real-world use would require additional security, privacy, validation, and operational safeguards.

## 👩‍💻 Author

**Urvashi Varshney**

GitHub: https://github.com/urvashi-varshney

## 📄 License

This project is intended for educational and research purposes. Add a suitable open-source license to the repository if you want others to reuse, modify, and distribute the code.

---

⭐ If you find this project useful, consider giving the repository a star on GitHub.
