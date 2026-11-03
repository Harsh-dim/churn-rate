import streamlit as st
import requests

api_url = "http://127.0.0.1:8000/predict"

st.title("Churn predictor")

st.markdown("Enter your details below : ")


gender =st.selectbox("gender (1 : male,0 : female)",options=[0,1])
SeniorCitizen = st.selectbox("SeniorCitizen (1: yes, 0: no)",options = [0,1])
Partner = st.selectbox("Partner (1: having, 0: no)",options = [0,1])
Dependents = st.selectbox("Dependents (1: yes, 0: no)",options = [0,1])
PhoneService = st.selectbox("PhoneService  (1: yes_avail, 0: no)",options = [0,1])
MultipleLines = st.selectbox("MultipleLines (1: yes, 0: no)",options = [0,1])
InternetService = st.selectbox("InternetService ('DSL', 'Fiber optic', 'No')",options = ['DSL', 'Fiber optic', 'No'])
OnlineSecurity = st.selectbox("OnlineSecurity (1: yes, 0: no)",options = [0,1])
OnlineBackup = st.selectbox("OnlineBackup (1: yes, 0: no)",options = [0,1])
DeviceProtection = st.selectbox("DeviceProtection (1: yes, 0: no)",options = [0,1])
TechSupport = st.selectbox("TechSupport (1: yes, 0: no)",options = [0,1])
StreamingTV = st.selectbox("StreamingTV (1: yes, 0: no)",options = [0,1])
StreamingMovies = st.selectbox("StreamingMovies  (1: yes, 0: no)",options = [0,1])
PaperlessBilling = st.selectbox("PaperlessBilling (1: yes, 0: no)",options = [0,1])
tenure = st.number_input("tenure",min_value = 0,value =5,max_value = 99)
MonthlyCharges = st.number_input("MonthlyCharges",min_value = 10,value =50,max_value = 1000)
PaymentMethod = st.selectbox("Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'",options = ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
Contract = st.selectbox("'Month-to-month', 'One year', 'Two year'",options = ['Month-to-month', 'One year', 'Two year'])

if st.button("predict churn"):
    input_data = {
              
                'gender' : gender,
                'SeniorCitizen' : SeniorCitizen,
                'Partner' : Partner,
                'Dependents' : Dependents,
                'tenure' : tenure,
                'PhoneService': PhoneService,
                'MultipleLines' : MultipleLines,
                'InternetService': InternetService,
                "OnlineSecurity": OnlineSecurity,
                'OnlineBackup' : OnlineBackup,
                'DeviceProtection': DeviceProtection,
                'TechSupport': TechSupport,
                'StreamingTV': StreamingTV,
                'StreamingMovies' : StreamingMovies,
                'Contract': Contract,
                'PaperlessBilling': PaperlessBilling,
                'PaymentMethod': PaymentMethod,
                'MonthlyCharges' : MonthlyCharges
    }

    try : 
        response = requests.post(api_url,json = input_data)
        if response.status_code == 200 :
            result = response.json()
            st.success(f"your prediction is{result}")
        else:
            st.error(f"api_error : {response.status_code} - {response.text}" )
    except requests.exceptions.ConnectionError:
        st.error("request not worked properly")
