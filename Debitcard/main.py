import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn import metrics

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


data=pd.read_csv("D:\\Fast\\\python\\project\\debitcard\\card.csv")
print(data.head())
print(data.info())
print(data.isnull().sum())
print(data.describe())
print(data.shape())

legit=data[data.Class==0]
fraud=data[data.Class==1]
print(legit.shape)
print(fraud.shape)


legit_sample=legit.sample(n=492)
new_data=pd.concat([legit_sample, fraud], axis=0)

print(new_data['Class'].value_counts())

X=new_data.drop("Class",axis=1)
y=new_data["Class"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,stratify=y ,random_state=2)

model=LogisticRegression()
model.fit(X_train, y_train)
test_prediction=model.predict(X_test)
print("Logistic Regression Model trained successfully.")
accu=accuracy_score(test_prediction, y_test)
print("Logistic Regression Accuracy:", accu)

input_data=()

numasarray=np.asarray(input_data).reshape(1,-1)
prediction=model.predict(numasarray)
print("Prediction for input data:", prediction)
