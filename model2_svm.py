import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

df = pd.read_csv('data/TCS_stock_history.csv')

df['HL_Perc'] = (df['High'] - df['Low']) / df['Low']
df['OC_Perc'] = (df['Close'] - df['Open']) / df['Open']
df['Volatility'] = df['HL_Perc'].rolling(window=10).mean()
df['VolumeChange'] = df['Volume'].pct_change()

prediction_window = 100
df['PriceNextMonth'] = df['Close'].shift(-prediction_window)
df.dropna(inplace=True)

X = df[['Open', 'High', 'Low', 'Volume','Close']].values
X = preprocessing.scale(X)
y = df['PriceNextMonth'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = svm.SVR(kernel='rbf', C=100, gamma='scale', epsilon=0.1)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
accuracy = model.score(X_test, y_test)
print(f"Mean Squared Error: {mse}")
print("accuracy:" ,accuracy)

plt.scatter(range(len(y_test)), y_test, c='b', label="Actual", alpha=0.7)
plt.scatter(range(len(y_test)), y_pred, c='r', label="Predicted", alpha=0.7)
plt.xlabel("Data Point")
plt.ylabel("Price")
plt.title("Stock Price Prediction with SVM")
plt.legend(loc='best')
plt.show()
