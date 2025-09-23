from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel


app = FastAPI()

model = joblib.load("calorie_model.pkl")

@app.get("/")
def home():
    return {"message": "Welcome to the Calorie Burn Prediction API"}

class Features(BaseModel):
    Gender: str
    Age: int
    Height_m: float
    Weight_kg: float
    BMI: float
    Fat_Percentage: float
    Water_Intake_liters: float
    Workout_Type: str
    Experience_Level: str
    Workout_Frequency_daysperweek: int
    Session_Duration_hours: float
    Resting_BPM: int
    Avg_BPM: int
    Max_BPM: int

'Experience_Level'

@app.post("/predict")
def predict(features: Features):
    input_data = pd.DataFrame([features.model_dump()])
    prediction = model.predict(input_data)[0]
    return {"calories_burned": round(float(prediction), 2)}