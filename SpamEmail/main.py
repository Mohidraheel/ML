import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
 
 
df = pd.read_csv("D:\\Fast\\python\\project\\spam.csv", encoding='latin-1')
df.columns = ['label', 'message']

df['label']=df['label'].map({'ham':0 , 'spam':1})
print(df.isnull().sum())
X=df['message']
y=df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer= TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)


nb=MultinomialNB()
nb.fit(X_train_vectorized, y_train)
nb_prediction = nb.predict(X_test_vectorized)

print("Naive Bayes Model trained successfully.")
print("Naive Bayes Prediction:", nb_prediction)
print("Naive Bayes Accuracy:", accuracy_score(y_test, nb_prediction))
print(confusion_matrix(y_test, nb_prediction))
print(classification_report(y_test, nb_prediction))

lr=LogisticRegression()
lr.fit(X_train_vectorized, y_train)
lr_prediction = lr.predict(X_test_vectorized)

print("Logistic Regression Model trained successfully.")
print("Logistic Regression Prediction:", lr_prediction)
print("Logistic Regression Accuracy:", accuracy_score(y_test, lr_prediction))
print(confusion_matrix(y_test, lr_prediction))
print(classification_report(y_test, lr_prediction))
