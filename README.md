# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the packages.

2.Analyse the data.

3.Use modelselection and Countvectorizer to preditct the values.

4.Find the accuracy and display the result.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: Lokeshwaran S
RegisterNumber:212224240080
*/

import pandas as pd
data=pd.read_csv("spam.csv", encoding='Windows-1252')
data

data.shape

x=data['v2'].values
y=data['v1'].values
x.shape

y.shape

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2, random_state=0)
x_train

x_train.shape

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
acc=accuracy_score(y_test,y_pred)
acc

con=confusion_matrix(y_test,y_pred)
print(con)

cl=classification_report(y_test,y_pred)
print(cl)
```

## Output:

## data

<img width="1048" height="657" alt="image" src="https://github.com/user-attachments/assets/526160f9-b7b3-40ed-bf0c-d4a77ee9cc08" />

## confusion matrix

<img width="962" height="83" alt="image" src="https://github.com/user-attachments/assets/60e0c2d1-2200-48ac-8219-ab98bc81f345" />

## accuracy

<img width="877" height="77" alt="image" src="https://github.com/user-attachments/assets/34aea191-8f20-4b51-b86d-6142d27e5097" />

## classification report

<img width="873" height="265" alt="image" src="https://github.com/user-attachments/assets/7c7a1434-2aaf-4c8d-aebb-7f697af579fc" />


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
