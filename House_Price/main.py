import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Load the dataset
house_data = pd.read_csv('D:\\Fast\\python\\project\\housepredictor\\Bengaluru_House_Data.csv')


house_data = house_data.dropna()


house_data.drop(columns=["area_type", "society", "balcony", "availability"], inplace=True)


def convert_sqft_to_number(x):
    try:
        if '-' in x:
            parts = x.split('-')
            return (float(parts[0]) + float(parts[1])) / 2
        else:
            return float(x)
    except:
        return None 


house_data['total_sqft'] = house_data['total_sqft'].apply(convert_sqft_to_number)


house_data = house_data.dropna(subset=['total_sqft'])
print(house_data['total_sqft'].head(10))


print(house_data.corr(numeric_only=True)["price"])
sns.lmplot(x="total_sqft", y="price", data=house_data, fit_reg=True, ci=None)
plt.show()
predictors=["total_sqft", "bath"]
target="price"
train=house_data[house_data["price"]<150].copy()
test=house_data[house_data["price"]>=150].copy()
print("Train size:", train.shape)
print("Test size:", test.shape)
print(test.head())


reg=LinearRegression()
reg.fit(train[predictors], train[target])
LinearRegression()
prediction=reg.predict(test[predictors])
test["predicted_price"] = prediction
print(test.head(10))
test.loc[test["predicted_price"] < 0, "predicted_price"] = 0
test["predicted_price"] = test["predicted_price"].round()
error = mean_absolute_error(test[target], test["predicted_price"])
print(f"Mean Absolute Error: {error}")
