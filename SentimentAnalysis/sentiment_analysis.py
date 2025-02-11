import numpy as np
import pandas as pd
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_explore_data():
    """
    Loads the IMDB Movie Reviews dataset and performs basic exploration.
    :return: Features (X) and target labels (y)
    """
    # Load the dataset
    try:
        reviews = load_files("data/aclImdb", categories=['pos', 'neg'], encoding='utf-8')
        X = reviews.data
        y = reviews.target

        if len(X) == 0:
            raise ValueError("No samples found in the dataset. Please verify the dataset location.")

        print("\nDataset Information:")
        print(f"Number of samples: {len(X)}")
        print(f"Target classes: {reviews.target_names}")
        print(f"Shape of the dataset: ({len(X)}, 1)")

        return X, y
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Please ensure the dataset is downloaded and placed in the 'data/aclImdb' folder.")
        exit(1)


def preprocess_data(X, y):
    """
    Splits the data into training and testing sets and vectorizes the text data.
    :param X: Text data
    :param y: Target labels
    :return: Vectorized training and testing data
    """
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Vectorize the text data using TF-IDF
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    return X_train_vec, X_test_vec, y_train, y_test


def train_model(X_train, y_train):
    """
    Trains a Logistic Regression model on the training data.
    :param X_train: Vectorized training features
    :param y_train: Training labels
    :return: Trained model
    """
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluates the trained model on the testing data.
    :param model: Trained model
    :param X_test: Vectorized testing features
    :param y_test: Testing labels
    """
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {accuracy:.2f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()


if __name__ == "__main__":
    print("Welcome to the Sentiment Analysis Project! 🎬")

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