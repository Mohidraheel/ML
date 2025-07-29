import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error


teams = pd.read_csv('D:\\Fast\\python\\project\\p1\\teams.csv', usecols=["team", "country", "year", "athletes", "age", "prev_medals", "medals"])
# print(teams)


print(teams.corr(numeric_only=True)["medals"])

sns.lmplot(x="prev_medals", y="medals", data=teams, fit_reg=True,ci=None)
# plt.show()

teams[teams.isnull().any(axis=1)]
teams=teams.dropna()

# print(teams)


train=teams[teams["year"]<2012].copy()
test=teams[teams["year"]>=2012].copy()
predictors=["age","prev_medals"]
target="medals"

reg=LinearRegression()
reg.fit(train[predictors], train[target])
LinearRegression()
prediction=reg.predict(test[predictors])
test["predicted_medals"]=prediction
test.loc[test["predicted_medals"]<0, "predicted_medals"]=0
test["predicted_medals"]=test["predicted_medals"].round()

error=mean_absolute_error(test[target], test["predicted_medals"])
print(f"Mean Absolute Error: {error}")
print(teams.describe()["medals"])
print(test[test["team"]=="USA"])

