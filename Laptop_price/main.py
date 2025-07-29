import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_absolute_error
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

data=pd.read_csv("D:\\Fast\\python\\project\\laptop\\laptop_data.csv")
data['Ram'] = data['Ram'].str.replace('GB', '').astype(int)
data['Weight'] = data['Weight'].str.replace('kg', '').astype(float)
data.drop(columns=["Unnamed: 0"],inplace=True)
print(data.head())
sns.lmplot(x="Inches", y="Price", data=data, fit_reg=True, ci=None)
plt.show()
new=data['ScreenResolution'].str.split('x',n=1, expand=True)
x_res= new[0]
y_res= new[1]
data['x_res'] = x_res
data['y_res'] = y_res
data['x_res'] = data['x_res'].str.replace(' ', '').str.findall(r'(\d+\.?\d+)'  ).apply(lambda x:x[0])
data['x_res'] = data['x_res'].astype(int)
data['y_res']=data['y_res'].astype(int)
data['Touchscreen'] = data['ScreenResolution'].str.contains('Touchscreen').astype(int)
data['IPS'] = data['ScreenResolution'].str.contains('IPS').astype(int)
data['ppi']= np.sqrt(data['x_res']**2 + data['y_res']**2) / (data['Inches']).astype(float)

data.drop(columns=["ScreenResolution", "x_res", "y_res","Inches"], inplace=True)


data["Cpu Name"]=data['Cpu'].apply(lambda x:" ".join(x.split()[0:3]))


def extract_cpu_brand(cpu_name):
    if cpu_name == "Intel Core i7" or cpu_name == "Intel Core i5" or cpu_name == "Intel Core i3":
        return cpu_name
    else:
        if cpu_name.split()[0] == "Intel":
            return "Intel Other"
        else:
            return "AMD"
data["Cpu Brand"] = data["Cpu Name"].apply(extract_cpu_brand)
data.drop(columns=["Cpu", "Cpu Name"], inplace=True)

data["Gpu Brand"] = data['Gpu'].apply(lambda x: x.split()[0])
data["Gpu Brand"] = data["Gpu Brand"].apply(lambda x: "Nvidia" if "Nvidia" in x else ("AMD" if "AMD" in x else "Intel" if "Intel" in x else "Other"))
data.drop(columns=["Gpu"], inplace=True)  # ← Add this line

# Convert 'Memory' column values to strings
data["Memory"] = data["Memory"].astype(str)
data["Memory"] = data["Memory"].str.replace("GB", "", regex=False)
data["Memory"] = data["Memory"].str.replace("TB", "000", regex=False)

def extract_hdd(memory_str):
    parts = memory_str.split('+')
    total = 0
    for part in parts:
        if "HDD" in part:
            num = ''.join(filter(str.isdigit, part))
            if num:
                total += int(num)
    return total

def extract_ssd(memory_str):
    parts = memory_str.split('+')
    total = 0
    for part in parts:
        if "SSD" in part:
            num = ''.join(filter(str.isdigit, part))
            if num:
                total += int(num)
    return total

data["HDD"] = data["Memory"].apply(extract_hdd)
data["SSD"] = data["Memory"].apply(extract_ssd)
data.drop(columns=["Memory"], inplace=True)

print(data.corr(numeric_only=True)["Price"])
X=data.drop(columns=["Price"])
Y=np.log(data["Price"])

print(data.head())
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15, random_state=2)

categorical_cols = ['Company', 'TypeName', 'OpSys', 'Cpu Brand', 'Gpu Brand']

step1 = ColumnTransformer([
    ('col_tnf', OneHotEncoder(sparse_output=False, drop='first'), categorical_cols)
], remainder='passthrough')


step2=LinearRegression()
Pipe = Pipeline([
    ('step1', step1),
    ('step2', step2)
])
Pipe.fit(X_train, Y_train)
prediction = Pipe.predict(X_test)
print(f"R2 Score: {r2_score(Y_test, prediction)}")
print(f"Mean Absolute Error: {mean_absolute_error(Y_test, prediction)}")
