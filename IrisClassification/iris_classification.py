import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_explore_data():
    """
    Loads the Iris dataset and performs basic exploration.
    :return: Features (X) and target labels (y)
    """
    # Load the Iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names

    # Convert to a DataFrame for better visualization
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
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the features for better performance
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test


def train_model(X_train, y_train):
    """
    Trains a Logistic Regression model on the training data.
    :param X_train: Scaled training features
    :param y_train: Training labels
    :return: Trained model
    """
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluates the trained model on the testing data.
    :param model: Trained model
    :param X_test: Scaled testing features
    :param y_test: Testing labels
    """
    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {accuracy:.2f}")

    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Setosa', 'Versicolor', 'Virginica'],
                yticklabels=['Setosa', 'Versicolor', 'Virginica'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()


if __name__ == "__main__":
    print("Welcome to the Iris Dataset Classification! 🌸")

    # Step 1: Load and explore the data
    X, y = load_and_explore_data()

    # Step 2: Preprocess the data
    X_train, X_test, y_train, y_test = preprocess_data(X, y)

    # Step 3: Train the model
    print("\nTraining the Logistic Regression model...")
    model = train_model(X_train, y_train)

    # Step 4: Evaluate the model
    print("\nEvaluating the model...")
    evaluate_model(model, X_test, y_test)