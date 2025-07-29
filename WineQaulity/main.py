import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


data=pd.read_csv("D:\\Fast\\python\\project\\wineqaulity\\winequalityred.csv")

print(data.head())
print(data.corr(numeric_only=True)["quality"])

print(data.isnull().sum())

sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.show()
X=data.drop("quality",axis=1)
print(X)

Y=data["quality"].apply(lambda y:1 if y>=7 else 0)
print(Y)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=3)

model=RandomForestClassifier()
model.fit(X_train,y_train)

test_prediction=model.predict(X_test)
print("Random Forest Model trained successfully.")
test_accu=accuracy_score(test_prediction,y_test)
print("Random Forest Accuracy:", test_accu)

input_data=(7.3, 0.65, 0, 1.2, 0.065, 15, 21, 0.9946, 3.39, 0.47, 10)

numpyarray=np.asarray(input_data)
input_data_reshaped=numpyarray.reshape(1,-1)
prediction=model.predict(input_data_reshaped)
print("Prediction for input data:", prediction)


