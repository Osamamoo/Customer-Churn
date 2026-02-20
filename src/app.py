# import streamlit as st
# import pandas as pd
# from predict import predict_single
# import plotly.graph_objects as go
# import os

# st.set_page_config(page_title="Customer Churn Prediction", page_icon="📊", layout="wide")

# st.markdown("""
#     <style>
#     .main-header {font-size: 2.5rem; font-weight: bold; text-align: center; color: #2c3e50; margin-bottom: 1.5rem;}
#     .prediction-box {padding: 2rem; border-radius: 10px; text-align: center; margin: 2rem 0;}
#     .churn-yes {background-color: #fee; border: 2px solid #e74c3c;}
#     .churn-no {background-color: #efe; border: 2px solid #27ae60;}
#     </style>
# """, unsafe_allow_html=True)

# st.markdown('<p class="main-header">Customer Churn Prediction System</p>', unsafe_allow_html=True)
# st.markdown("---")

# st.sidebar.title("Navigation")
# page = st.sidebar.radio("Select Page", ["Single Prediction", "Batch Prediction"])

# MODEL_PATH = "models/adaboost_model.pkl"
# if not os.path.exists(MODEL_PATH):
#     st.error("Model not found! Please train the model first by running: python model.py")
#     st.stop()

# if page == "Single Prediction":
#     st.header("Single Customer Prediction")
    
#     col1, col2, col3 = st.columns(3)
    
#     with col1:
#         st.subheader("Demographics")
#         gender = st.selectbox("Gender", ["Male", "Female"])
#         senior_citizen = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
#         partner = st.selectbox("Partner", ["Yes", "No"])
#         dependents = st.selectbox("Dependents", ["Yes", "No"])
    
#     with col2:
#         st.subheader("Services")
#         phone_service = st.selectbox("Phone Service", ["Yes", "No"])
#         multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
#         internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
#         online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
#         online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
#         device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
#         tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
#         streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
#         streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
    
#     with col3:
#         st.subheader("Account Information")
#         tenure = st.slider("Tenure (months)", 0, 72, 12)
#         contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
#         paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
#         payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
#         monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=70.0, step=0.5)
#         total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=350.0, step=10.0)
    
#     if st.button("Predict Churn", type="primary", use_container_width=True):
#         input_data = {
#             "gender": gender, "SeniorCitizen": senior_citizen, "Partner": partner,
#             "Dependents": dependents, "tenure": tenure, "PhoneService": phone_service,
#             "MultipleLines": multiple_lines, "InternetService": internet_service,
#             "OnlineSecurity": online_security, "OnlineBackup": online_backup,
#             "DeviceProtection": device_protection, "TechSupport": tech_support,
#             "StreamingTV": streaming_tv, "StreamingMovies": streaming_movies,
#             "Contract": contract, "PaperlessBilling": paperless_billing,
#             "PaymentMethod": payment_method, "MonthlyCharges": monthly_charges,
#             "TotalCharges": str(total_charges)
#         }
        
#         try:
#             result = predict_single(input_data)
#             prediction = result["prediction"]
#             probability = result["probability"]
            
#             st.markdown("---")
#             st.subheader("Prediction Results")
            
#             col_res1, col_res2 = st.columns(2)
            
#             with col_res1:
#                 if prediction == 1:
#                     st.markdown(f"""
#                         <div class="prediction-box churn-yes">
#                             <h2>CHURN</h2>
#                             <p style="font-size: 1.5rem; margin-top: 1rem;">
#                                 Churn Probability: <strong>{probability * 100:.2f}%</strong>
#                             </p>
#                         </div>
#                     """, unsafe_allow_html=True)
#                 else:
#                     st.markdown(f"""
#                         <div class="prediction-box churn-no">
#                             <h2>NOT CHURN</h2>
#                             <p style="font-size: 1.5rem; margin-top: 1rem;">
#                                 Churn Probability: <strong>{probability * 100:.2f}%</strong>
#                             </p>
#                         </div>
#                     """, unsafe_allow_html=True)
            
#             with col_res2:
#                 fig = go.Figure(go.Indicator(
#                     mode="gauge+number",
#                     value=probability * 100,
#                     domain={'x': [0, 1], 'y': [0, 1]},
#                     title={'text': "Churn Risk Score"},
#                     gauge={
#                         'axis': {'range': [None, 100]},
#                         'bar': {'color': "#2c3e50"},
#                         'steps': [
#                             {'range': [0, 30], 'color': "#27ae60"},
#                             {'range': [30, 70], 'color': "#f39c12"},
#                             {'range': [70, 100], 'color': "#e74c3c"}
#                         ],
#                         'threshold': {'line': {'color': "#c0392b", 'width': 4}, 'thickness': 0.75, 'value': 50}
#                     }
#                 ))
#                 fig.update_layout(height=300)
#                 st.plotly_chart(fig, use_container_width=True)
            
#             st.markdown("---")
#             st.subheader("Recommendations")
#             if prediction == 1:
#                 st.warning("Immediate retention campaign recommended. Consider personal outreach and loyalty offers.")
#             else:
#                 st.success("Continue current engagement strategy and monitor usage patterns.")
        
#         except Exception as e:
#             st.error(f"Error making prediction: {str(e)}")

# else:
#     st.header("Batch Prediction")
    
#     uploaded_file = st.file_uploader("Upload CSV file with customer data", type="csv")
    
#     if uploaded_file is not None:
#         try:
#             df = pd.read_csv(uploaded_file)
#             st.write("Preview of uploaded data:")
#             st.dataframe(df.head())
            
#             if st.button("Run Batch Prediction", type="primary"):
#                 with st.spinner("Making predictions..."):
#                     predictions = []
#                     probabilities = []
                    
#                     for idx, row in df.iterrows():
#                         result = predict_single(row.to_dict())
#                         predictions.append(result["prediction"])
#                         probabilities.append(result["probability"])
                    
#                     df["Churn_Prediction"] = predictions
#                     df["Churn_Probability"] = probabilities
#                     df["Risk_Level"] = df["Churn_Probability"].apply(
#                         lambda x: "High" if x > 0.7 else "Medium" if x > 0.3 else "Low"
#                     )
                    
#                     st.success("Predictions completed!")
                    
#                     col1, col2, col3 = st.columns(3)
#                     with col1:
#                         st.metric("Total Customers", len(df))
#                     with col2:
#                         st.metric("Predicted Churns", sum(predictions))
#                     with col3:
#                         st.metric("Churn Rate", f"{(sum(predictions) / len(df)) * 100:.2f}%")
                    
#                     st.write("Prediction Results:")
#                     st.dataframe(df)
                    
#                     csv = df.to_csv(index=False)
#                     st.download_button("Download Results as CSV", data=csv, file_name="churn_predictions.csv", mime="text/csv")
                    
#                     risk_counts = df["Risk_Level"].value_counts()
#                     fig = go.Figure(data=[go.Pie(
#                         labels=risk_counts.index,
#                         values=risk_counts.values,
#                         hole=.3,
#                         marker_colors=['#e74c3c', '#f39c12', '#27ae60']
#                     )])
#                     fig.update_layout(title="Risk Level Distribution")
#                     st.plotly_chart(fig, use_container_width=True)
        
#         except Exception as e:
#             st.error(f"Error processing file: {str(e)}")






import streamlit as st
import pandas as pd
from predict import predict_single
import plotly.graph_objects as go
import os

st.set_page_config(page_title="Customer Churn Prediction", page_icon="📊", layout="wide")

st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem; font-weight: bold; text-align: center;
        color: #2c3e50; margin-bottom: 1.5rem;
    }
    .prediction-box {
        padding: 2rem; border-radius: 12px; text-align: center;
        margin: 2rem 0; color: #2c3e50; font-weight: 600;
    }
    .churn-yes {
        background: linear-gradient(135deg, #ffcccc, #ff9999);
        border: 2px solid #e74c3c;
    }
    .churn-no {
        background: linear-gradient(135deg, #ccffcc, #99e699);
        border: 2px solid #27ae60;
    }
    .stMetric {
        background-color: #f8f9fa; border-radius: 10px;
        padding: 10px; box-shadow: 0 2px 6px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">Customer Churn Prediction System</p>', unsafe_allow_html=True)
st.markdown("---")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Page", ["Single Prediction", "Batch Prediction"])

MODEL_PATH = "models/adaboost_model.pkl"
if not os.path.exists(MODEL_PATH):
    st.error("Model not found! Please train the model first by running: python model.py")
    st.stop()

if page == "Single Prediction":
    st.header("Single Customer Prediction")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Demographics")
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior_citizen = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
    
    with col2:
        st.subheader("Services")
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
    
    with col3:
        st.subheader("Account Information")
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=70.0, step=0.5)
        total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=350.0, step=10.0)
    
    if st.button("Predict Churn", type="primary", use_container_width=True):
        input_data = {
            "gender": gender, "SeniorCitizen": senior_citizen, "Partner": partner,
            "Dependents": dependents, "tenure": tenure, "PhoneService": phone_service,
            "MultipleLines": multiple_lines, "InternetService": internet_service,
            "OnlineSecurity": online_security, "OnlineBackup": online_backup,
            "DeviceProtection": device_protection, "TechSupport": tech_support,
            "StreamingTV": streaming_tv, "StreamingMovies": streaming_movies,
            "Contract": contract, "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method, "MonthlyCharges": monthly_charges,
            "TotalCharges": str(total_charges)
        }
        
        try:
            result = predict_single(input_data)
            prediction = result["prediction"]
            probability = result["probability"]
            
            st.markdown("---")
            st.subheader("Prediction Results")
            
            col_res1, col_res2 = st.columns(2)
            
            with col_res1:
                if prediction == 1:
                    st.markdown(f"""
                        <div class="prediction-box churn-yes">
                            <h2>CHURN LIKELY</h2>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                        <div class="prediction-box churn-no">
                            <h2>NOT CHURN</h2>
                        </div>
                    """, unsafe_allow_html=True)
            
            with col_res2:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=probability * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Churn Risk Score"},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "#34495e"},
                        'steps': [
                            {'range': [0, 30], 'color': "#2ecc71"},
                            {'range': [30, 70], 'color': "#f1c40f"},
                            {'range': [70, 100], 'color': "#e74c3c"}
                        ],
                        'threshold': {'line': {'color': "#c0392b", 'width': 4}, 'thickness': 0.75, 'value': 50}
                    }
                ))
                fig.update_layout(height=300, paper_bgcolor="#f8f9fa", font={'color': "#2c3e50"})
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            st.subheader("Recommendations")
            if prediction == 1:
                st.warning("Immediate retention campaign recommended. Consider personal outreach and loyalty offers.")
            else:
                st.success("Continue current engagement strategy and monitor usage patterns.")
        
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")

else:
    st.header("Batch Prediction")
    
    uploaded_file = st.file_uploader("Upload CSV file with customer data", type="csv")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("Preview of uploaded data:")
            st.dataframe(df.head())
            
            if st.button("Run Batch Prediction", type="primary"):
                with st.spinner("Making predictions..."):
                    predictions = []
                    probabilities = []
                    
                    for idx, row in df.iterrows():
                        result = predict_single(row.to_dict())
                        predictions.append(result["prediction"])
                        probabilities.append(result["probability"])
                    
                    df["Churn_Prediction"] = predictions
                    df["Churn_Probability"] = probabilities
                    df["Risk_Level"] = df["Churn_Probability"].apply(
                        lambda x: "High" if x > 0.7 else "Medium" if x > 0.3 else "Low"
                    )
                    
                    st.success("Predictions completed")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Customers", len(df))
                    with col2:
                        st.metric("Predicted Churns", sum(predictions))
                    with col3:
                        st.metric("Churn Rate", f"{(sum(predictions) / len(df)) * 100:.2f}%")
                    
                    st.write("Prediction Results:")
                    st.dataframe(df)
                    
                    csv = df.to_csv(index=False)
                    st.download_button("Download Results as CSV", data=csv, file_name="churn_predictions.csv", mime="text/csv")
                    
                    risk_counts = df["Risk_Level"].value_counts()
                    fig = go.Figure(data=[go.Pie(
                        labels=risk_counts.index,
                        values=risk_counts.values,
                        hole=.4,
                        marker_colors=['#e74c3c', '#f1c40f', '#2ecc71']
                    )])
                    fig.update_layout(title="Risk Level Distribution", paper_bgcolor="#f8f9fa", font={'color': "#2c3e50"})
                    st.plotly_chart(fig, use_container_width=True)
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
