# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required packages and print the present data.

2.Print the placement data and salary data.

3.Find the null and duplicate values.

4.Using logistic regression find the predicted values of accuracy , confusion matrices.

5.Display the results.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: JANSI RANI A A
RegisterNumber:  2122240404130

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
data = pd.read_csv("Placement_Data.csv")
print(data.head())

# Copy and drop unnecessary columns
data1 = data.copy()
data1 = data1.drop(["sl_no", "salary"], axis=1)
print(data1.head())

# Check nulls and duplicates
print(data1.isnull().sum())
print("Duplicates:", data1.duplicated().sum())

# Label encoding
le = LabelEncoder()
data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])

# Features and target
x = data1.iloc[:, :-1]
y = data1["status"]

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=0
)

# Logistic Regression
lr = LogisticRegression(solver="liblinear")
lr.fit(x_train, y_train)

# Predictions
y_pred = lr.predict(x_test)
print("Predictions:", y_pred)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Classification report
classification_report1 = classification_report(y_test, y_pred)
print(classification_report1)

# Predict for a sample input
sample_pred = lr.predict([[1, 80, 1, 90, 1, 1, 90, 1, 0, 85, 1, 85]])
print("Sample Prediction:", sample_pred)


*/
```

## Output:

<img width="711" height="140" alt="Screenshot (731)" src="https://github.com/user-attachments/assets/f1868424-fa24-467e-b2c1-bbf5bc3bef77" />

<img width="707" height="142" alt="Screenshot (732)" src="https://github.com/user-attachments/assets/080e66e3-5712-40a4-a9fb-2c5b449bd755" />

<img width="729" height="151" alt="Screenshot (733)" src="https://github.com/user-attachments/assets/fbefd612-5ee2-4389-b6df-1fa49826008d" />

<img width="538" height="145" alt="Screenshot (734)" src="https://github.com/user-attachments/assets/a2bce0ec-0fae-4a5b-8d6d-fb4d9083050b" />

<img width="587" height="43" alt="Screenshot (737)" src="https://github.com/user-attachments/assets/4c1e2f80-d450-4bb4-9443-c6e66917b52d" />

<img width="602" height="209" alt="Screenshot (736)" src="https://github.com/user-attachments/assets/634febff-bed2-4f9e-947d-6f872b20a791" />

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
