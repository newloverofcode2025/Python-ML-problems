import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

def load_and_explore_data(ticker="AAPL", start="2015-01-01", end="2023-01-01"):
    """
    Loads and explores historical stock price data.
    :param ticker: Stock ticker symbol (e.g., 'AAPL' for Apple)
    :param start: Start date for the data
    :param end: End date for the data
    :return: DataFrame containing the stock data
    """
    # Load the data
    stock_data = yf.download(ticker, start=start, end=end)

    print("\nFirst 5 rows of the dataset:")
    print(stock_data.head())

    print("\nDataset Information:")
    print(f"Shape of the dataset: {stock_data.shape}")

    return stock_data


def preprocess_data(stock_data):
    """
    Preprocesses the data by creating features and target variables.
    :param stock_data: Original DataFrame
    :return: Features (X) and target labels (y)
    """
    # Use the 'Close' column as the target variable
    stock_data['Target'] = stock_data['Close'].shift(-1)  # Predict the next day's closing price

    # Drop rows with NaN values (last row will have NaN for 'Target')
    stock_data = stock_data[:-1]

    # Use the 'Close' price as the feature
    X = np.array(stock_data[['Close']])
    y = np.array(stock_data['Target'])

    return X, y


def train_model(X_train, y_train):
    """
    Trains a Linear Regression model on the training data.
    :param X_train: Training features
    :param y_train: Training labels
    :return: Trained model
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluates the trained model on the testing data.
    :param model: Trained model
    :param X_test: Testing features
    :param y_test: Testing labels
    """
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"\nModel Evaluation:")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

    # Plot the actual vs predicted prices
    plt.figure(figsize=(10, 6))
    plt.plot(y_test, label='Actual Prices', color='blue')
    plt.plot(y_pred, label='Predicted Prices', color='red')
    plt.title('Actual vs Predicted Stock Prices')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    print("Welcome to the Stock Price Prediction Project! 📈")

    # Step 1: Load and explore the data
    stock_data = load_and_explore_data()

    # Step 2: Preprocess the data
    X, y = preprocess_data(stock_data)

    # Step 3: Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 4: Train the model
    print("\nTraining the Linear Regression model...")
    model = train_model(X_train, y_train)

    # Step 5: Evaluate the model
    print("\nEvaluating the model...")
    evaluate_model(model, X_test, y_test)