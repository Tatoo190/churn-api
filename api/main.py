import joblib
import pandas as pd

from fastapi import FastAPI
app = FastAPI()
model = joblib.load(
    "../models/churn_model.pkl"
)
from pydantic import BaseModel
class Customer(BaseModel):
    credit_score:int
    age:int
    tenure:int
    balance:float
    products_number:int
    credit_card:int
    active_member:int
    estimated_salary:float
    country_Germany:int
    country_Spain:int
    gender_Male:int


@app.post("/predict")
def predict(customer: Customer):
    data = pd.DataFrame(
    [customer.model_dump()]
    )

    prediction = model.predict(
    data)[0]

    probability = model.predict_proba(
    data)[0][1]

    return {
    "churn_prediction": int(prediction),
    "churn_probability": round(float(probability),4)
}