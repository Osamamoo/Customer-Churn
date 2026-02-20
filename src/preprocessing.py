import pandas as pd
import joblib
import os

EXPECTED_COLUMNS_PATH = "models/expected_columns.pkl"

def load_data(path: str) -> pd.DataFrame:
    """Load dataset from CSV file."""
    return pd.read_csv(path)

def preprocess(df: pd.DataFrame, save_expected: bool = False) -> pd.DataFrame:
    """
    Clean and preprocess the Telco Customer Churn dataset.

    Args:
        df: Input DataFrame (raw dataset or raw user input).
        save_expected: If True, save the list of expected columns after one-hot encoding
                       (used during training). At inference, the function aligns columns
                       to this list.
    Returns:
        Preprocessed DataFrame ready for modeling.
    """
    data = df.copy()

    # Convert TotalCharges to numeric (coerce errors to NaN)
    if "TotalCharges" in data.columns:
        data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")

    # Drop missing values (training only)
    if "Churn" in data.columns:  
        data.dropna(inplace=True)

    # Drop unneeded columns (ignore if not present)
    data.drop(["customerID", "Partner", "PhoneService"], axis=1, errors="ignore", inplace=True)

    # Encode binary columns (only those that exist)
    binary_cols = ["Dependents", "PaperlessBilling", "Churn"]
    for col in binary_cols:
        if col in data.columns:
            data[col] = data[col].map({"Yes": 1, "No": 0})

    # Encode gender if available
    if "gender" in data.columns:
        data["gender"] = data["gender"].map({"Female": 0, "Male": 1})

    # One-hot encode nominal categorical columns (only those that exist)
    nominal_cols = [
        "Contract", "MultipleLines", "InternetService", "OnlineSecurity",
        "OnlineBackup", "DeviceProtection", "TechSupport",
        "StreamingTV", "StreamingMovies", "PaymentMethod"
    ]
    existing_nominals = [col for col in nominal_cols if col in data.columns]
    data = pd.get_dummies(data, columns=existing_nominals, drop_first=False)

    # Save expected columns during training
    if save_expected:
        joblib.dump(list(data.columns), EXPECTED_COLUMNS_PATH)

    # At inference: align columns to training schema
    if not save_expected and os.path.exists(EXPECTED_COLUMNS_PATH):
        expected_cols = joblib.load(EXPECTED_COLUMNS_PATH)
        for col in expected_cols:
            if col not in data.columns:
                data[col] = 0
        # Ensure same column order
        data = data[expected_cols]

    return data