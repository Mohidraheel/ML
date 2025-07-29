import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score, mean_absolute_error,accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn import svm


df=pd.read_csv("D:\\Fast\\python\\project\\diabetics\\diabetes.csv")
print(df.groupby('Outcome').mean())
X=df.drop(columns='Outcome',axis=1)
Y=df['Outcome']
print(X)
print(Y)

scalar=StandardScaler()
scalar.fit(X)
standadized_data=scalar.fit_transform(X)

X=standadized_data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2,stratify=Y ,random_state=2)

model=svm.SVC(kernel='linear')
model.fit(X_train, Y_train)
print("Model trained successfully.")
X_train_prediction=model.predict(X_train)
X_train_accuracy=accuracy_score(X_train_prediction,Y_train)
print("Accuracy on training data:", X_train_accuracy)

X_test_prediction=model.predict(X_test)
X_test_accuracy=accuracy_score(X_test_prediction,Y_test)
print("Accuracy on test data:", X_test_accuracy)
