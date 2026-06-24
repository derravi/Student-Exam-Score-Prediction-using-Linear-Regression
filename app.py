import pandas as pd
from fastapi import FastAPI
import pickle
from Schema.pydentic_mode import UserInput
from fastapi.responses import JSONResponse

app = FastAPI(title="Student Exam Score Prediction using Linear Regression")

with open('models/model.pkl', 'rb') as f:
    loaded_data = pickle.load(f)

best_mdl = loaded_data['bst_model']
best_accuracy = loaded_data['bst_accuracy']

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
    
    prediction_output = round(float((best_mdl.predict(df))[0]),2)

    semester_average = (predict.Semester1 + predict.Semester2 + predict.Semester3 + predict.Semester4) / 4

    # Category generate
    if prediction_output >= 85:
        category = "Excellent"
    elif prediction_output >= 70:
        category = "Good"
    elif prediction_output >= 50:
        category = "Average"
    else:
        category = "At Risk"

    #Recommendations Code
    recommendations = []

    if predict.StudyHours < 4:
        recommendations.append(
            "Increase daily study hours to at least 4-5 hours."
        )

    if predict.PreviousMarks < 60:
        recommendations.append(
            "Strengthen fundamental concepts and revise previous topics."
        )

    if semester_average < 65:
        recommendations.append(
            "Revise previous semester subjects regularly."
        )

    if prediction_output < 50:
        recommendations.extend([
            "Seek additional academic support and mentoring.",
            "Solve more practice papers and mock tests."
        ])

    elif prediction_output < 70:
        recommendations.append(
            "Maintain consistency in studies and improve weak subjects."
        )

    elif prediction_output < 85:
        recommendations.append(
            "Good performance. Focus on advanced practice and revision."
        )

    else:
        recommendations.extend([
            "Excellent performance!",
            "Maintain your current study routine.",
            "Challenge yourself with advanced topics and competitive exams."
        ])


    return JSONResponse(status_code=200,content={
        'Prediction':{
            "The Total Predicted Marks of the student is:":prediction_output,
            "With the Accuracy of the model is " : f"{round(best_accuracy,2)*100}%",
        },
        'academic_report':{
            "average_semester_marks": round(semester_average, 2),
            "performance_category": category
        },
        "recommendations": recommendations
    })