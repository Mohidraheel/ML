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


data=pd.read_csv("D:\\Fast\\python\\project\\heartdisease\\heart_disease_data.csv")

print(data)
print(data.corr(numeric_only=True)["target"])
print(data.isnull().sum())
sns.heatmap(data.select_dtypes(include=np.number).corr(), annot=True, cmap='coolwarm')
plt.show()
print(data.describe())
X=data.drop(columns="target",axis=1)
print(X)
y=data["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,stratify=y ,random_state=2)

model=LogisticRegression()
model.fit(X_train, y_train)

test_prediction=model.predict(X_test)
print("Logistic Regression Model trained successfully.")
accu=accuracy_score(test_prediction,y_test)
print("Logistic Regression Accuracy:", accu)

input_data=(37,1,2,130,250,0,1,187,0,3.5,0,0,2)

numpyasarray=np.asarray(input_data).reshape(1,-1)
prediction=model.predict(numpyasarray)
print("Heart Disease Prediction:", prediction)
