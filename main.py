from fastapi import FastAPI
import pandas as pd
import joblib

app = FastAPI()

@app.get("/")
def root():
    return {"mensagem": "Bem-vindo à API de Machine Learning!"}

@app.get("/dados")
def get_dados(estado: str = None):
    df = pd.read_csv("dados.csv")
    if estado:
        df = df[df['estado'] == estado]
    return df.to_dict(orient="records")

@app.post("/predict")
def predict(data: dict):
    modelo = joblib.load("modelo.pkl")
    X = pd.DataFrame([data])
    pred = modelo.predict(X)
    return {"classe": int(pred[0])}
