import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

def load_and_explore_data():
    """
    Loads the Iris dataset and performs basic exploration.
    :return: Features (X) and target labels (y)
    """
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names

    df = pd.DataFrame(X, columns=feature_names)
    df['species'] = y

    print("First 5 rows of the dataset:")
    print(df.head())

    print("\nDataset Information:")
    print(f"Features: {feature_names}")
    print(f"Target Classes: {target_names}")
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


def train_model_with_tuning(X_train, y_train, model_type="logistic"):
    """
    Trains a classification model with hyperparameter tuning using GridSearchCV.
    :param X_train: Scaled training features
    :param y_train: Training labels
    :param model_type: Type of model to train ('logistic', 'tree', 'forest', 'knn')
    :return: Best model after tuning
    """
    if model_type == "logistic":
        model = LogisticRegression(max_iter=1000)
        param_grid = {"C": [0.1, 1, 10], "solver": ["liblinear"]}
    elif model_type == "tree":
        model = DecisionTreeClassifier(random_state=42)
        param_grid = {"max_depth": [3, 5, 10], "min_samples_split": [2, 5, 10]}
    elif model_type == "forest":
        model = RandomForestClassifier(random_state=42)
        param_grid = {"n_estimators": [50, 100, 200], "max_depth": [3, 5, 10]}
    elif model_type == "knn":
        model = KNeighborsClassifier()
        param_grid = {"n_neighbors": [3, 5, 7], "weights": ["uniform", "distance"]}
    else:
        raise ValueError("Invalid model type. Choose from 'logistic', 'tree', 'forest', 'knn'.")

    grid_search = GridSearchCV(model, param_grid, cv=5, scoring="accuracy")
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
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {accuracy:.2f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Setosa', 'Versicolor', 'Virginica'],
                yticklabels=['Setosa', 'Versicolor', 'Virginica'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()


def save_model(model, filename="model.pkl"):
    """
    Saves the trained model to a file.
    :param model: Trained model
    :param filename: Name of the file to save the model
    """
    joblib.dump(model, filename)
    print(f"Model saved to {filename}")


if __name__ == "__main__":
    print("Welcome to the Iris Dataset Classification! 🌸")

    # Step 1: Load and explore the data
    X, y = load_and_explore_data()

    # Step 2: Preprocess the data
    X_train, X_test, y_train, y_test = preprocess_data(X, y)

    # Step 3: Train the model with hyperparameter tuning
    model_type = input("Enter the model type ('logistic', 'tree', 'forest', 'knn'): ").strip().lower()
    model = train_model_with_tuning(X_train, y_train, model_type)

    # Step 4: Evaluate the model
    evaluate_model(model, X_test, y_test)

    # Step 5: Save the model
    save_model(model)