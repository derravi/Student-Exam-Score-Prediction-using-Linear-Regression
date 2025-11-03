import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score
import matplotlib.pyplot as plt
import numpy as np
import pickle

#Load Dataset
print("Lets inport the Datasets........\n")
df = pd.read_csv("Data set/student_dataset.csv")

print("Now we are select the fetures for the Model predictions.............\n")
#Input and output.
fetures = ["Age","Attendance","StudyHours","PreviousMarks","SleepHours","Semester1","Semester2","Semester3","Semester4"]
x = df[fetures]
y = df[["FinalMarks"]]



print("Lets Train the Linear Regression Model for this datasets.................\n")
#Train Model 
model = LinearRegression()
model.fit(x,y)
predicted_score = model.predict(x)

print("Lets see the Different Different Parameters for this Machine Learning Models.................\n")
#Velid Regression Metrics
mae = mean_absolute_error(y,predicted_score)
mse = mean_squared_error(y,predicted_score)
rmse = np.sqrt(mse)
r2 = r2_score(y,predicted_score)

#Show Result
#The round(name,2) is roundupt he long floating values into 2 decimal after the dot.
print("Mean Absolute Error(MEA):",round(mae,2))
print("Mean Squared Error(MSE):",round(mse,2))
print("Root Mean Squared Error(RMSE):",round(rmse,2))
print("R^2 Score (Model Accuracy):",round(r2,2))

print("User Input Fields.........\n")

user_data = []

for i in fetures:
    value = float(input(f"Enter the {i}:"))
    user_data.append(value)

user_data = pd.DataFrame([user_data])

final_prediction = model.predict(user_data)

print(f"The Final Prediction output is : {round(final_prediction[0][0],2)}%")

print("Lets make the pickle model for this projects.............\n")

with open('models/model.pkl','wb') as f:
    pickle.dump(model,f)

