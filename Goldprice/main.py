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


data=pd.read_csv("D:\\Fast\\python\\project\\goldprice\\gld_price_data.csv")

print(data.head())
print(data.info())
print(data.isnull().sum())
print(data.corr(numeric_only=True,)["GLD"])
sns.heatmap(data.select_dtypes(include=np.number).corr(), annot=True, cmap='coolwarm')
plt.show()
X=data.drop(["Date", "GLD"], axis=1)
Y=data["GLD"]
print(X)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=3)

model=RandomForestRegressor(n_estimators=100)

model.fit(X_train, y_train)

test_prediction=model.predict(X_test)
print("Random Forest Model trained successfully.")

error=metrics.r2_score(y_test, test_prediction)
print("Random Forest R^2 Score:", error)

# ploting the actual and prediction value

plt.plot(y_test.values, label='Actual', color='blue')
plt.plot(test_prediction, label='Predicted', color='orange')
plt.title('Actual vs Predicted Gold Prices')
plt.xlabel('Sample Index')
plt.ylabel('Gold Price')
plt.legend()
plt.show()
