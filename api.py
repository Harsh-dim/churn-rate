from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel,Field,computed_field
from typing import Literal,Annotated
import joblib
import pandas as pd

model = joblib.load(r"C:\Users\HARSH\OneDrive\Desktop\churn_rate_model\Churn_model.pkl")
app =FastAPI()

class UserInput(BaseModel):
    gender: Annotated[Literal[0, 1], Field(..., description="1: male, 0: female")]
    SeniorCitizen :  Annotated[Literal[0, 1], Field(..., description="1: yes, 0: no")]
    Partner : Annotated[Literal[0, 1], Field(..., description="1: yes, 0: no")]
    Dependents : Annotated[Literal[0, 1], Field(..., description="1: yes, 0: no")]
    PhoneService : Annotated[Literal[0, 1], Field(..., description="1: yes, 0: no")]
    MultipleLines : Annotated[Literal[0, 1], Field(..., description="1: yes, 0: no")]
    InternetService : Annotated[Literal['DSL', 'Fiber optic', 'No'], Field(...)]
    OnlineSecurity : Annotated[Literal[0, 1], Field(..., description="1: yes, 0: no")]
    OnlineBackup : Annotated[Literal[0, 1], Field(..., description="1: yes, 0: no")]
    DeviceProtection : Annotated[Literal[0, 1], Field(..., description="1: yes, 0: no")]
    TechSupport : Annotated[Literal[0, 1], Field(..., description="1: yes, 0: no")]
    StreamingTV : Annotated[Literal[0, 1], Field(..., description="1: yes, 0: no")]
    StreamingMovies : Annotated[Literal[0, 1], Field(..., description="1: yes, 0: no")]
    PaperlessBilling : Annotated[Literal[0, 1], Field(..., description="1: yes, 0: no")]
    tenure : Annotated[int, Field(...,lt = 100)]
    MonthlyCharges : Annotated[float, Field(...,gt = 10,lt = 500)]
    PaymentMethod : Annotated[Literal['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'], Field(...)]
    Contract : Annotated[Literal['Month-to-month', 'One year', 'Two year'], Field(...)]

    @computed_field
    @property
    def TotalCharges(self) -> float:
        return self.tenure * self.MonthlyCharges
    
@app.post('/predict')
def predict_churn(data : UserInput):

    input_df = pd.DataFrame([{
        'TotalCharges' : data.TotalCharges,
        'gender' : data.gender,
        'SeniorCitizen' : data.SeniorCitizen,
        'Partner' : data.Partner,
        'Dependents' : data.Dependents,
        'tenure' : data.tenure,
        'PhoneService': data.PhoneService,
        'MultipleLines' : data.MultipleLines,
        'InternetService': data.InternetService,
        "OnlineSecurity": data.OnlineSecurity,
        'OnlineBackup' : data.OnlineBackup,
        'DeviceProtection': data.DeviceProtection,
        'TechSupport': data.TechSupport,
        'StreamingTV': data.StreamingTV,
        'StreamingMovies' : data.StreamingMovies,
        'Contract': data.Contract,
        'PaperlessBilling': data.PaperlessBilling,
        'PaymentMethod': data.PaymentMethod,
        'MonthlyCharges' : data.MonthlyCharges

    }])

    pred = int(model.predict(input_df)[0])
    proba = float(model.predict_proba(input_df)[0][1])

    return JSONResponse(
        status_code=200,
        content={
            'predicted_category': pred,
            'churn_label': 'Will churn' if pred == 1 else 'Will not churn',
            'churn_probability': round(proba, 3)
        }
        )




    
