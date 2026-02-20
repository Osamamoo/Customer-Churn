import joblib
import pandas as pd
from preprocessing import preprocess

# Load model and scaler
MODEL_PATH = "models/adaboost_model.pkl"
saved_artifacts = joblib.load(MODEL_PATH)
model = saved_artifacts["model"]
scaler = saved_artifacts["scaler"]

def predict_single(raw_input: dict) -> dict:
    """
    raw_input: dict with raw customer data (before preprocessing).
    Example:
    {
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 5,
        "PhoneService": "No",
        "Contract": "One year",
        "MonthlyCharges": 70.5,
        "TotalCharges": "350.5",
        "PaymentMethod": "Electronic check",
        ...
    }
    """

    # 1. Convert to DataFrame
    df = pd.DataFrame([raw_input])

    # 2. Apply preprocessing (same as training)
    df_processed = preprocess(df)

    # Ensure target column is not present
    if "Churn" in df_processed.columns:
        df_processed = df_processed.drop("Churn", axis=1)

    # 3. Apply scaling
    df_scaled = scaler.transform(df_processed)

    # 4. Make prediction
    prediction = model.predict(df_scaled)[0]
    probability = model.predict_proba(df_scaled)[0][1]

    return {
        "prediction": int(prediction),         # 0 = No Churn, 1 = Churn
        "probability": round(probability, 4)  # Probability of churn
    }


if __name__ == "__main__":
    # Example test
    sample = {
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 5,
        "PhoneService": "No",
        "MultipleLines": "No phone service",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "No",
        "DeviceProtection": "Yes",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "No",
        "Contract": "One year",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70.5,
        "TotalCharges": "350.5"
    }

    result = predict_single(sample)
    print(result)