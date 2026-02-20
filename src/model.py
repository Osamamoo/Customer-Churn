import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from imblearn.combine import SMOTEENN
from sklearn.ensemble import AdaBoostClassifier
import joblib

from preprocessing import load_data, preprocess


def train_model(data_path: str, save_path: str = "models/adaboost_model.pkl"):
    # ===== 1. Load & preprocess data =====
    df = load_data(data_path)
    df = preprocess(df, save_expected=True)

    # ===== 2. Separate features and target =====
    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    # ===== 3. Train/Test split =====
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=15, stratify=y
    )

    # ===== 4. Handle imbalance with SMOTEENN =====
    sm = SMOTEENN()
    X_res, y_res = sm.fit_resample(X, y)

    # Split again after resampling
    Xr_train, Xr_test, yr_train, yr_test = train_test_split(
        X_res, y_res, test_size=0.2, random_state=42
    )

    # ===== 5. Scaling =====
    scaler = MinMaxScaler()
    Xr_train_scaled = scaler.fit_transform(Xr_train)
    Xr_test_scaled = scaler.transform(Xr_test)

    # ===== 6. Define & train AdaBoost =====
    model = AdaBoostClassifier(
        learning_rate=0.5, 
        n_estimators=165, 
        random_state=42
    )
    model.fit(Xr_train_scaled, yr_train)

    # ===== 7. Evaluation =====
    y_train_pred = model.predict(Xr_train_scaled)
    y_test_pred = model.predict(Xr_test_scaled)

    print("Train Accuracy:", np.round(accuracy_score(yr_train, y_train_pred)*100, 2))
    print("Test Accuracy:", np.round(accuracy_score(yr_test, y_test_pred)*100, 2))
    print("\nConfusion Matrix:\n", confusion_matrix(yr_test, y_test_pred))
    print("\nClassification Report:\n", classification_report(yr_test, y_test_pred))

    # ===== 8. Save model & scaler =====
    joblib.dump({"model": model, "scaler": scaler}, save_path)
    print(f"\nModel saved to {save_path}")


if __name__ == "__main__":
    train_model("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")