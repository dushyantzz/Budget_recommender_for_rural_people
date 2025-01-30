from fastapi import FastAPI
import pickle
import pandas as pd

# Import the script if you need functions or classes defined there
#import budget_recommender_for_rural_people

app = FastAPI()

# Load your model
with open("best_model.pkl", "rb") as f:
    best_model = pickle.load(f)

@app.get("/")
def read_root():
    return {"message": "Budget Recommender API for Rural People is running!"}

@app.post("/predict")
def predict_overspending(income: float, groceries: float, utilities: float,
                         transportation: float, healthcare: float,
                         entertainment: float, savings: float):
    # Construct a single-row DataFrame from the input data
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

    # Return a message indicating overspending vs. within budget
    result = "Overspending" if prediction == 1 else "Within Budget"
    return {"prediction": result}
