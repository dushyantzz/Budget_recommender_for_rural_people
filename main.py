from fastapi import FastAPI
import pickle
import pandas as pd


app = FastAPI()

with open("best_model.pkl", "rb") as f:
    best_model = pickle.load(f)

@app.get("/")
def read_root():
    return {"message": "Budget Recommender API for Rural People is running!"}

@app.post("/predict")
def predict_overspending(income: float, groceries: float, utilities: float,
                         transportation: float, healthcare: float,
                         entertainment: float, savings: float):
    df_input = pd.DataFrame([{
        "Income": income,
        "Groceries": groceries,
        "Utilities": utilities,
        "Transportation": transportation,
        "Healthcare": healthcare,
        "Entertainment": entertainment,
        "Savings": savings
    }])

    prediction = best_model.predict(df_input)[0]

    result = "Overspending" if prediction == 1 else "Within Budget"
    return {"prediction": result}
