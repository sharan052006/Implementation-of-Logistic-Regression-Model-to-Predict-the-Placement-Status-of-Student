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

5.Display the results

## Program:

Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

Developed by: Sharan.I

RegisterNumber:  212224040308

```python
import pandas as pd
data=pd.read_csv("Placement_Data.csv")
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)#Browses the specified row or column
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"] )     
data1["status"]=le.fit_transform(data1["status"])       
data1

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```

## Output:
DATA HEAD:

![{B854F338-8FF3-4F66-BE4A-1FF777F7F01C}](https://github.com/user-attachments/assets/553122f9-1ab7-4cf6-b6a1-6cf45e073090)

DATA1 HEAD:

![{33C9D393-C09A-4CA2-BBD0-B986CA770E4D}](https://github.com/user-attachments/assets/4ccd1092-b82a-4e2c-8df6-52c52e53f685)

ISNULL().SUM():

![{FA909CDF-A48A-423A-841F-F0FB9DEC5EFB}](https://github.com/user-attachments/assets/6bc3f8a6-9ced-45fa-8fcd-aea7ad486bf6)

DATA DUPLICATE:

![{897C4808-70F4-4ACF-A8CF-A9D2EC8FE474}](https://github.com/user-attachments/assets/88165dc9-2ad8-48f5-ba01-5eb5edec0217)

PRINT DATA:

![{DF1A04EA-6A04-4890-9882-4F4770417FD8}](https://github.com/user-attachments/assets/d3f8800b-d675-427a-9225-142bd233015d)

STATUS:

![{D4F359C3-4624-4AA3-9558-0FA5A344F4FE}](https://github.com/user-attachments/assets/38eac62b-f3a4-4b88-a5a9-3bb8ddb25b29)

Y_PRED:

![{1565805F-C906-4EAE-A5CE-ACC8398ABE21}](https://github.com/user-attachments/assets/76e270ad-9053-43e7-beac-54b62d1c6e09)

ACCURACY:

![{A4A1A61D-7AD8-482B-AA78-8059B1E94385}](https://github.com/user-attachments/assets/06876911-9e62-4bdf-85ef-356783a85523)

CONFUSION MATRIX:

![{82909DB7-B682-486E-B36C-FFB261F7A49E}](https://github.com/user-attachments/assets/6d77c855-3da3-4790-8e4d-5ebbd07806c1)

CLASSIFICATION:

![{BB0D4A7B-0611-40B0-A8EE-61E290F2B834}](https://github.com/user-attachments/assets/5b2fa0f8-dd54-42dc-93c5-1e16660c8435)

LR PREDICT:

![{5B68C0DC-86FF-4AB1-A7CC-A8439B542AE9}](https://github.com/user-attachments/assets/56c26aaa-6837-4d55-b97d-8b421289c906)



## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
