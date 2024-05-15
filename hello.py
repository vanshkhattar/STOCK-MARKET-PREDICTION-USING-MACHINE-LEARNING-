def model1():
    import pandas as pd
    import numpy as np
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    import matplotlib.pyplot as plt
    from matplotlib import style
    import sklearn
    from sklearn.preprocessing import StandardScaler

    #style for the plot
    style.use("dark_background")

    df = pd.read_csv('E:/TCS_stock_history.csv')

    # two new columns for feature engineering
    df["HL_Perc"] = (df["High"] - df["Low"]) / df["Low"] * 100
    df["CO_Perc"] = (df["Close"] - df["Open"]) / df["Open"] * 100

    #date array
    dates = np.array(df["Date"])
    dates_check = dates[-30:]
    dates = dates[:-30]

    #features and target variable
    df = df[["Open", "High", "Low", "Close"]]
    df["PriceNextMonth"] = df["Close"].shift(-30)

    # Feature scaling
    scaler = StandardScaler()
    X = np.array(df.drop(["PriceNextMonth"], axis=1))
    X = scaler.fit_transform(X)
    X_Check = X[-30:]
    X = X[:-30]

    df.dropna(inplace=True)
    y = np.array(df["PriceNextMonth"])

    # Splitting the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #random forest regression model
    model = RandomForestRegressor()

    # Fiting the model using the training data
    model.fit(X_train, y_train)

    # Calculating the accuracy
    acc = model.score(X_test, y_test)
    print("accuracy:", acc)

    # model fitting
    model.fit(X, y)

    # Makeing predictions
    predictions = model.predict(X_Check)

    # Calculateing mean squared error
    mse = mean_squared_error(y_test, model.predict(X_test))
    print(f"Mean Squared Error: {mse}")

    #DataFrames for actual and forecast values
    actual = pd.DataFrame(dates, columns=["Date"])
    actual["ClosePrice"] = df["Close"]
    actual["Forecast"] = np.nan
    actual.set_index("Date", inplace=True)

    forecast = pd.DataFrame(dates_check, columns=["Date"])
    forecast["Forecast"] = predictions
    forecast["ClosePrice"] = np.nan
    forecast.set_index("Date", inplace=True)

    # Concatenate actual and forecast dataframes
    result = pd.concat([actual, forecast])

    # Plotting the results
    plt.plot(result.index, result["ClosePrice"], c='b', label="Actual Close Price")
    plt.plot(result.index, result["Forecast"], c='r', label="Forecasted Close Price")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.title("Stock Price Prediction with Random Forest")
    plt.legend(loc='best')
    plt.show()




def model2():
    import numpy as np
    import pandas as pd
    from sklearn import preprocessing
    from sklearn.model_selection import train_test_split
    from sklearn import svm
    import matplotlib.pyplot as plt
    from sklearn.metrics import mean_squared_error

    # stock price data from a CSV file
    df = pd.read_csv('E:/TCS_stock_history.csv')

    # Feature engineering
    df['HL_Perc'] = (df['High'] - df['Low']) / df['Low']
    df['OC_Perc'] = (df['Close'] - df['Open']) / df['Open']
    df['Volatility'] = df['HL_Perc'].rolling(window=10).mean()
    df['VolumeChange'] = df['Volume'].pct_change()

    #  prediction window
    prediction_window = 100

    # Create target variable
    df['PriceNextMonth'] = df['Close'].shift(-prediction_window)

    # Drop NaN values
    df.dropna(inplace=True)

    #features and target variable
    X = df[['Open', 'High', 'Low', 'Volume','Close']].values
    X = preprocessing.scale(X)
    y = df['PriceNextMonth'].values

    #  data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    #moldel train
    model = svm.SVR(kernel='rbf', C=100, gamma='scale', epsilon=0.1)
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)
    # Calculate mean square error and accuracy
    mse = mean_squared_error(y_test, y_pred)
    accuracy = model.score(X_test, y_test)
    print(f"Mean Squared Error: {mse}")
    print("accuracy:" ,accuracy)
    #graph plot
    plt.scatter(range(len(y_test)), y_test, c='b', label="Actual", alpha=0.7)
    plt.scatter(range(len(y_test)), y_pred, c='r', label="Predicted", alpha=0.7)
    plt.xlabel("Data Point")
    plt.ylabel("Price")
    plt.title("Stock Price Prediction with SVM")
    plt.legend(loc='best')
    plt.show()




def model3():
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.model_selection import train_test_split
    from keras.models import Sequential
    from keras.layers import Dense
    import matplotlib.pyplot as plt

    # Load the stock price data from a CSV file
    df = pd.read_csv('E:/TCS_stock_history.csv')
    data = df['Close'].values.reshape(-1, 1)

    # Normalize the data using Min-Max scaling
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)

    # Create input features and target variable
    X, y = [], []
    look_back = 30  # Number of previous time steps to use for prediction

    for i in range(len(data) - look_back):
        X.append(data[i : i + look_back, 0])
        y.append(data[i + look_back, 0])

    X, y = np.array(X), np.array(y)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build the ANN model
    model = Sequential()
    model.add(Dense(64, input_dim=look_back, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='linear'))

    # Compile the model
    model.compile(loss='mean_squared_error', optimizer='adam')

    # Train the model
    model.fit(X_train, y_train, epochs=50, batch_size=32)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Inverse transform the predictions to the original scale
    y_test = y_test.reshape(-1, 1)
    y_pred = scaler.inverse_transform(y_pred)
    y_test = scaler.inverse_transform(y_test)
    accuracy = model.evaluate(X_test, y_test)
    print("accuracy:" ,accuracy)
    # Plot actual vs. predicted prices
    plt.plot(y_test, label='Actual Prices', color='blue')
    plt.plot(y_pred, label='Predicted Prices', color='red')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()


