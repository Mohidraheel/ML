import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score, mean_absolute_error,accuracy_score

df=pd.read_csv("D:\\Fast\\python\\project\\sonardata\\Copy of sonar data.csv")
print(df.isnull().sum())
print(df.groupby(df.columns[60]).mean())
X = df.drop(columns=['R'])  
Y = df['R']                 

print(X)
print(Y)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1,stratify=Y, random_state=1)
model=LogisticRegression()
model.fit(X_train, y_train)

train_pred=model.predict(X_train)
train_accu=accuracy_score(train_pred, y_train)

print("Training Accuracy:", train_accu)

test_pred = model.predict(X_test)
test_accu=accuracy_score(test_pred, y_test)

print("Testing Accuracy:", test_accu)

input_data = (
    0.02, 0.0371, 0.0428, 0.0207, 0.0954, 0.0986, 0.1539, 0.1601, 0.3109, 0.2111,
    0.1609, 0.1582, 0.2238, 0.0645, 0.066, 0.2273, 0.31, 0.2999, 0.5078, 0.4797,
    0.5783, 0.5071, 0.4328, 0.555, 0.6711, 0.6415, 0.7104, 0.808, 0.6791, 0.3857,
    0.1307, 0.2604, 0.5121, 0.7547, 0.8537, 0.8507, 0.6692, 0.6097, 0.4943, 0.2744,
    0.051, 0.2834, 0.2825, 0.4256, 0.2641, 0.1386, 0.1051, 0.1343, 0.0383, 0.0324,
    0.0232, 0.0027, 0.0065, 0.0159, 0.0072, 0.0167, 0.018, 0.0084, 0.009, 0.0032
)
inputasnumpy= np.asarray(input_data).reshape(1, -1)
prediction = model.predict(inputasnumpy)
print("Prediction:", prediction)
