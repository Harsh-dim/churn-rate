# Customer Churn Prediction: Problem Statement

## 1. Background

Customer churn, when subscribers stop using a service, is one of the costliest problems for subscription businesses such as telecom providers. Acquiring a new customer is usually far more expensive than retaining an existing one, so a company that can identify likely churners *before* they leave can target them with offers, support, or better plans and protect its revenue.

This project builds a machine learning system that predicts whether a telecom customer is likely to churn, and serves that prediction through a REST API.

## 2. Problem Definition

**Goal:** Given a customer's demographics, subscribed services, contract and billing details, predict whether the customer will churn (`1`) or stay (`0`).

**Task type:** Supervised binary classification.

**Business objective:** Catch as many true churners as possible (high **recall**) while keeping the false-alarm rate reasonable (acceptable **precision**), so retention campaigns reach the right customers.

## 3. Dataset

- **Source:** IBM Telco Customer Churn dataset (`WA_Fn-UseC_-Telco-Customer-Churn.csv`)
- **Size:** 7,043 customers, 21 columns (20 features + target `Churn`)
- **Target distribution:** 5,174 stayed (No) vs 1,869 churned (Yes), about 26.5% churn, so the classes are **imbalanced**.
- **Feature groups:**
  - *Demographics:* gender, SeniorCitizen, Partner, Dependents
  - *Services:* PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies
  - *Account and billing:* tenure, Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges

## 4. Key Challenges

1. **Class imbalance:** Churners are a minority, so a naive model can score high accuracy while missing most churners.
2. **Messy data:** `TotalCharges` contains blank strings (for brand-new customers) that must be cleaned before it can be used as a number.
3. **Mixed feature types:** The data has binary, ordinal (Contract), nominal (InternetService, PaymentMethod) and skewed numeric features, each needing different preprocessing.
4. **Overfitting:** Tree-based models reached a train AUC of 1.0 with a much lower CV AUC, so they needed regularisation and tuning.

## 5. Approach

1. **Data cleaning and EDA:** Dropped `customerID`, fixed blank `TotalCharges`, and studied distributions, skewness, outliers and churn rate by category.
2. **Feature encoding:**
   - Yes/No columns (and "No internet/phone service") mapped to 0/1; gender mapped to 0/1
   - `Contract` ordinal-encoded (Month-to-month < One year < Two year)
   - `InternetService` and `PaymentMethod` one-hot encoded
   - `TotalCharges` log-transformed and scaled; `tenure` and `MonthlyCharges` scaled
3. **Imbalance handling:** SMOTE inside an `imblearn` pipeline, so oversampling is applied only to training folds and never leaks into validation or test data.
4. **Modelling:** Logistic Regression baseline, then Decision Tree, Random Forest and XGBoost compared with 5-fold stratified cross-validation.
5. **Tuning:** Random Forest hyperparameters tuned with Optuna (40 trials, optimising CV ROC-AUC).
6. **Deployment:** The full preprocessing + model pipeline is saved with `joblib` and served with **FastAPI**. Inputs are validated with **Pydantic** (`Literal` types and range checks), and `TotalCharges` is computed as `tenure × MonthlyCharges`.

## 6. Results

| Model | CV ROC-AUC | Notes |
|---|---|---|
| Decision Tree | 0.655 | Heavy overfitting (train AUC 1.0) |
| XGBoost | 0.811 | |
| Random Forest (default) | 0.817 | |
| **Random Forest (Optuna-tuned)** | **0.842** | Final model |

**Final model, held-out test set (30% split):** ROC-AUC **0.86**, Recall **0.76**, Precision **0.573**.

The Logistic Regression baseline reached a similar test AUC (0.86) with higher recall (0.82) but lower precision (0.54), so the tuned Random Forest offers a better precision/recall balance. The final choice can be adjusted depending on whether the business values catching every churner or limiting false alarms.

## 7. Deliverables

- Jupyter notebook covering EDA, preprocessing, model comparison and tuning
- Trained pipeline saved as `Churn_model.pkl`
- FastAPI service with a `/predict` endpoint that returns the predicted class, a readable label and the churn probability

## 8. Future Work

- Tune the decision threshold to hit a target recall or precision
- Add model explainability (feature importance / SHAP) to show *why* a customer is at risk
- Try cost-sensitive learning and further gradient-boosting tuning
- Containerise the API with Docker and deploy it to the cloud
- Monitor for data drift and retrain periodically
