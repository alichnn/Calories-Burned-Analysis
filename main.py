from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel


app = FastAPI()

model = joblib.load("calorie_model.pkl")

@app.get("/")
def home():
    return {"message": "Welcome to the Calorie Burn Prediction API"}

# Match training features exactly
class Features(BaseModel):
    Gender: str
    Age: int
    Height: float
    Weight: float
    BMI: float
    Fat_Percentage: float
    Water_Intake: float
    Workout_Type: str
    Experience_Level: str
    Workout_Frequency: int
    Session_Duration: float
    Resting_BPM: int
    Avg_BPM: int
    Max_BPM: int

@app.post("/predict")
def predict(features: Features):
    # Build DataFrame with exact column names
    input_data = pd.DataFrame([{
        "Gender": features.Gender,
        "Age": features.Age,
        "Height (m)": features.Height,
        "Weight (kg)": features.Weight,
        "BMI": features.BMI,
        "Fat_Percentage": features.Fat_Percentage,
        "Water_Intake (liters)": features.Water_Intake,
        "Workout_Type": features.Workout_Type,
        "Experience_Level": features.Experience_Level,
        "Workout_Frequency (days/week)": features.Workout_Frequency,
        "Session_Duration (hours)": features.Session_Duration,
        "Resting_BPM": features.Resting_BPM,
        "Avg_BPM": features.Avg_BPM,
        "Max_BPM": features.Max_BPM
    }])

    prediction = model.predict(input_data)[0]
    return {"calories_burned": round(float(prediction), 2)}