import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.preprocessing import StandardScaler

style.use("dark_background")

df = pd.read_csv('data/TCS_stock_history.csv')
df["HL_Perc"] = (df["High"] - df["Low"]) / df["Low"] * 100
df["CO_Perc"] = (df["Close"] - df["Open"]) / df["Open"] * 100

dates = np.array(df["Date"])
dates_check = dates[-30:]
dates = dates[:-30]

df = df[["Open", "High", "Low", "Close"]]
df["PriceNextMonth"] = df["Close"].shift(-30)

scaler = StandardScaler()
X = np.array(df.drop(["PriceNextMonth"], axis=1))
X = scaler.fit_transform(X)
X_Check = X[-30:]
X = X[:-30]

df.dropna(inplace=True)
y = np.array(df["PriceNextMonth"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor()
model.fit(X_train, y_train)

acc = model.score(X_test, y_test)
print("accuracy:", acc)

model.fit(X, y)
predictions = model.predict(X_Check)

mse = mean_squared_error(y_test, model.predict(X_test))
print(f"Mean Squared Error: {mse}")

actual = pd.DataFrame(dates, columns=["Date"])
actual["ClosePrice"] = df["Close"]
actual["Forecast"] = np.nan
actual.set_index("Date", inplace=True)

forecast = pd.DataFrame(dates_check, columns=["Date"])
forecast["Forecast"] = predictions
forecast["ClosePrice"] = np.nan
forecast.set_index("Date", inplace=True)

result = pd.concat([actual, forecast])

plt.plot(result.index, result["ClosePrice"], c='b', label="Actual Close Price")
plt.plot(result.index, result["Forecast"], c='r', label="Forecasted Close Price")
plt.xlabel("Date")
plt.ylabel("Price")
plt.title("Stock Price Prediction with Random Forest")
plt.legend(loc='best')
plt.show()
