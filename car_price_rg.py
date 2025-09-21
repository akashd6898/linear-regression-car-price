import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt
import requests
import os

from sklearn.metrics import mean_squared_error

url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/automobileEDA.csv"
file_name = "/Users/akashd/Downloads/usedcars.csv"

def fileaccess(url, file_name):
    response = requests.get(url)
    if(response.status_code == 200):
        with open(file_name, 'wb') as f:
            f.write(response.content)

if not os.path.exists(file_name):
    fileaccess(url, file_name)
else:
    print("File already exists, skipping... the download")

df = pd.read_csv(file_name)

print(df.head())

lm = LinearRegression()

X = df[['highway-mpg']]
Y = df['price']


lm.fit(X, Y)

Yhat=lm.predict(X)
print("mpg",Yhat[0:1])

#plt.scatter(X, Y, color='red')
#plt.plot(df['highway-mpg'],lm.predict(X), color='blue')
#plt.show()


# width = 12
# height = 10
# plt.figure(figsize=(width, height))
# sns.regplot(x="highway-mpg", y="price", data=df)
# plt.ylim(0,)
#plt.show()

X = df[['engine-size']]
Y = df['price']

lm.fit(X,Y)
Yhat=lm.predict(X)
print("engine-size",Yhat[0:1])

print("Engine Size r squared",lm.score(X, Y))

print("Engine Size: Intercept: when x is 0",lm.intercept_)
print("Engine Size: Co-efficient",lm.coef_)
print("Correlation of engine size", df[['engine-size','price']].corr())
print("MSE for Engine size price & source price data",mean_squared_error(Y,Yhat))
print("Average MSE per data point RMSE", np.sqrt(mean_squared_error(Y,Yhat)))

# width = 12
# height = 10
# plt.figure(figsize=(width, height))
# sns.regplot(x="engine-size", y="price", data=df)
# plt.ylim(0,)
#plt.show()


Z = df[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']]
lm.fit(Z, df['price'])
Yhat=lm.predict(Z)
print("multiple predictors",Yhat[0:1])

print("Intercept : when x is 0",lm.intercept_)
print("Co-efficient",lm.coef_)

print("Correlation of multiple predictors\n", df[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg', 'price']].corr())

print("Multiple predictors r squared",lm.score(Z, df['price']))

feature = ['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']
width = 12
height = 10
plt.figure(figsize=(width, height))
for col in feature:
    sns.regplot(x= col, y="price", data=df)
    plt.ylim(0,)
    plt.show()

print("MSE for Engine size price & source price data",mean_squared_error(df['price'],Yhat))
print("Average MSE per data point RMSE", np.sqrt(mean_squared_error(df['price'],Yhat)))

ax1 = sns.kdeplot(df['price'], color="r", label="Actual Value")
sns.kdeplot(Yhat, color="b", label="Fitted Values", ax=ax1)

plt.title('Actual vs Fitted Values for Price')
plt.xlabel('Price')
plt.ylabel('Proportion of Cars')
plt.legend()
plt.show()


#polylinear equation
def PlotPolly(model, independent_variable, dependent_variable, Name):
    x_new = np.linspace(independent_variable.min(),independent_variable.max(),100)
    y_new = model(x_new)

    plt.plot(independent_variable, dependent_variable, '.', x_new, y_new, '-')
    plt.title(f'Polynomial Fit for Price ~ {Name}')
    ax = plt.gca()
    ax.set_facecolor((0.898, 0.898, 0.898))
    fig = plt.gcf()
    plt.xlabel(Name)
    plt.ylabel('Price of Car')
    plt.show()

M = df[['engine-size']]
H = M.to_numpy().flatten()
Y = df['price']

f1 = np.polyfit(H, Y, 1)
p1 = np.poly1d(f1)
#predicted_with_p1 = np.polyval(p1, H)
print("polynomial of degree 1:\n",p1)
#print("polynomial of degree 1 price value:",predicted_with_p1)

f3 = np.polyfit(H, Y, 3)
p3 = np.poly1d(f3)
print("polynomial of degree 3:\n",p3)

f5 = np.polyfit(H, Y, 5)
p5 = np.poly1d(f5)
print("polynomial of degree 5:\n",p5)

PlotPolly(p1, M, Y, 'engine size')
PlotPolly(p3, M, Y, 'engine size')
PlotPolly(p5, M, Y, 'engine size')

r_squared_1 = r2_score(Y,p1(H))
r_squared_3 = r2_score(Y,p3(H))
print('The R-square value for 1st degree polynomial is: ', r_squared_1)
print('The MSE value for 1st degree polynomial is: ', mean_squared_error(Y,p1(H)))
print('The Root MSE for 1st degree polynomial is: ', np.sqrt(mean_squared_error(Y,p1(H))))
print('The R-square value for 3rd degree polynomial is: ', r_squared_3)
print('The MSE value for 3rd degree polynomial is: ', mean_squared_error(Y,p3(H)))
print('The Root MSE for 3rd degree polynomial is: ', np.sqrt(mean_squared_error(Y,p3(H))))
r_squared_5 = r2_score(Y,p5(H))
print('The R-square value for 5th degree polynomial is: ', r_squared_5)
print('The MSE value for 5th degree polynomial is: ', mean_squared_error(Y,p5(H)))
print('The Root MSE for 1st degree polynomial is: ', np.sqrt(mean_squared_error(Y,p5(H))))