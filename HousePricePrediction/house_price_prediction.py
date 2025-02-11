import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_explore_data():
    """
    Loads the California Housing dataset and performs basic exploration.
    :return: Features (X) and target labels (y)
    """
    housing = fetch_california_housing()
    X = housing.data
    y = housing.target
    feature_names = housing.feature_names

    df = pd.DataFrame(X, columns=feature_names)
    df['PRICE'] = y

    print("First 5 rows of the dataset:")
    print(df.head())

    print("\nDataset Information:")
    print(f"Features: {feature_names}")
    print(f"Shape of the dataset: {df.shape}")

    return X, y


def preprocess_data(X, y):
    """
    Splits the data into training and testing sets and scales the features.
    :param X: Features
    :param y: Target labels
    :return: Scaled training and testing data
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test


def train_model(X_train, y_train):
    """
    Trains a Linear Regression model on the training data.
    :param X_train: Scaled training features
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
    :param X_test: Scaled testing features
    :param y_test: Testing labels
    """
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"\nModel Evaluation:")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"R² Score: {r2:.2f}")


if __name__ == "__main__":
    print("Welcome to the House Price Prediction! 🏠")

    # Step 1: Load and explore the data
    X, y = load_and_explore_data()

    # Step 2: Preprocess the data
    X_train, X_test, y_train, y_test = preprocess_data(X, y)

    # Step 3: Train the model
    print("\nTraining the Linear Regression model...")
    model = train_model(X_train, y_train)

    # Step 4: Evaluate the model
    print("\nEvaluating the model...")
    evaluate_model(model, X_test, y_test)