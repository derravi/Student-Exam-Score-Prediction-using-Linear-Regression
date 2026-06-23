import pandas as pd
from fastapi import FastAPI
import pickle
from Schema.pydentic_mode import UserInput
from fastapi.responses import JSONResponse

app = FastAPI(title="Student Exam Score Prediction using Linear Regression")

with open('models/model.pkl', 'rb') as f:
    loaded_data = pickle.load(f)

rnd = loaded_data['Random_forest']

@app.get("/")
def default():
    return {"message":"Welcome to Student Exam Score Prediction using Linear Regression using Fast APIs"}

@app.post("/predict")
def predict_output(predict:UserInput):

    df = pd.DataFrame([{

        'StudyHours':predict.StudyHours,
        'PreviousMarks':predict.PreviousMarks,
        'Semester1':predict.Semester1,
        'Semester2':predict.Semester2,    
        'Semester3':predict.Semester3,
        'Semester4':predict.Semester3
    }])

    prediction_output = round(float((rnd.predict(df))[0]),2)

    return JSONResponse(status_code=200,content={
        "The Total Predicted Marks of the student is:":prediction_output
    })