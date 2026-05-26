import pickle
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

with open('model.bin', 'rb') as file:
    model = pickle.load(file)

numerical_columns = ['tenure', 'monthlycharges', 'totalcharges']
categorical_columns = [
    'gender', 'seniorcitizen', 'partner', 'dependents', 'phoneservice',
    'multiplelines', 'internetservice', 'onlinesecurity', 'onlinebackup',
    'deviceprotection', 'techsupport', 'streamingtv', 'streamingmovies',
    'contract', 'paperlessbilling', 'paymentmethod'
]


class Customer(BaseModel):
    customerid: str
    gender: str
    seniorcitizen: int
    partner: str
    dependents: str
    tenure: int
    phoneservice: str
    multiplelines: str
    internetservice: str
    onlinesecurity: str
    onlinebackup: str
    deviceprotection: str
    techsupport: str
    streamingtv: str
    streamingmovies: str
    contract: str
    paperlessbilling: str
    paymentmethod: str
    monthlycharges: float
    totalcharges: float


def prepare_data(customer_dict):
    df_small = pd.DataFrame([customer_dict])
    df_small['seniorcitizen'] = df_small['seniorcitizen'].astype('str')

    df_small_one_hot_encoded = pd.get_dummies(df_small[categorical_columns], dtype=int)
    df_final = pd.concat([df_small[numerical_columns], df_small_one_hot_encoded], axis=1)
    df_final = df_final.reindex(columns=model.feature_names_in_, fill_value=0)

    return df_final


@app.post("/predict")
def predict(customer: Customer):
    customer_prepared = prepare_data(customer.model_dump())
    churn_probability = model.predict_proba(customer_prepared)[:, 1][0]
    return {"churn_probability": round(float(churn_probability), 4)}
