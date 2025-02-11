import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

def load_and_explore_data():
    """
    Loads the Boston Housing dataset and performs basic exploration.
    :return: Features (X) and target labels (y)
    """
    boston = load_boston()
    X = boston.data
    y = boston.target
    feature_names = boston.feature_names

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


def train_model_with_tuning(X_train, y_train, model_type="linear"):
    """
    Trains a regression model with hyperparameter tuning using GridSearchCV.
    :param X_train: Scaled training features
    :param y_train: Training labels
    :param model_type: Type of model to train ('linear', 'tree', 'forest')
    :return: Best model after tuning
    """
    if model_type == "linear":
        model = LinearRegression()
        param_grid = {}  # No hyperparameters to tune for Linear Regression
    elif model_type == "tree":
        model = DecisionTreeRegressor(random_state=42)
        param_grid = {"max_depth": [3, 5, 10], "min_samples_split": [2, 5, 10]}
    elif model_type == "forest":
        model = RandomForestRegressor(random_state=42)
        param_grid = {"n_estimators": [50, 100, 200], "max_depth": [3, 5, 10]}
    else:
        raise ValueError("Invalid model type. Choose from 'linear', 'tree', 'forest'.")

    grid_search = GridSearchCV(model, param_grid, cv=5, scoring="neg_mean_squared_error")
    grid_search.fit(X_train, y_train)
    print(f"Best Parameters for {model_type}: {grid_search.best_params_}")
    return grid_search.best_estimator_


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


def save_model(model, filename="model.pkl"):
    """
    Saves the trained model to a file.
    :param model: Trained model
    :param filename: Name of the file to save the model
    """
    joblib.dump(model, filename)
    print(f"Model saved to {filename}")


if __name__ == "__main__":
    print("Welcome to the Boston Housing Price Prediction! 🏠")

    # Step 1: Load and explore the data
    X, y = load_and_explore_data()

    # Step 2: Preprocess the data
    X_train, X_test, y_train, y_test = preprocess_data(X, y)

    # Step 3: Train the model with hyperparameter tuning
    model_type = input("Enter the model type ('linear', 'tree', 'forest'): ").strip().lower()
    model = train_model_with_tuning(X_train, y_train, model_type)

    # Step 4: Evaluate the model
    evaluate_model(model, X_test, y_test)

    # Step 5: Save the model
    save_model(model)