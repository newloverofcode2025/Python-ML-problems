import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical

def load_and_explore_data():
    """
    Loads and explores the MNIST dataset.
    :return: Features (X) and target labels (y)
    """
    # Load the MNIST dataset
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    print("\nDataset Information:")
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Testing samples: {X_test.shape[0]}")
    print(f"Image shape: {X_train[0].shape}")
    print(f"Number of classes: {len(np.unique(y_train))}")

    # Display a sample image
    plt.figure(figsize=(6, 6))
    plt.imshow(X_train[0], cmap='gray')
    plt.title(f"Label: {y_train[0]}")
    plt.axis('off')
    plt.show()

    return X_train, X_test, y_train, y_test


def preprocess_data(X_train, X_test, y_train, y_test):
    """
    Preprocesses the data by scaling and reshaping.
    :param X_train: Training features
    :param X_test: Testing features
    :param y_train: Training labels
    :param y_test: Testing labels
    :return: Processed training and testing data
    """
    # Scale the pixel values to [0, 1]
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0

    # Reshape the data to include a channel dimension (for CNNs, if needed)
    X_train = X_train.reshape(-1, 28 * 28)
    X_test = X_test.reshape(-1, 28 * 28)

    # One-hot encode the labels
    y_train = to_categorical(y_train, num_classes=10)
    y_test = to_categorical(y_test, num_classes=10)

    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    """
    Trains a Neural Network model on the training data.
    :param X_train: Training features
    :param y_train: Training labels
    :return: Trained model
    """
    model = Sequential([
        Dense(128, activation='relu', input_shape=(28 * 28,)),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.2)

    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluates the trained model on the testing data.
    :param model: Trained model
    :param X_test: Testing features
    :param y_test: Testing labels
    """
    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"\nModel Accuracy: {accuracy:.2f}")

    # Make predictions
    y_pred = np.argmax(model.predict(X_test), axis=1)
    y_true = np.argmax(y_test, axis=1)

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    # Classification Report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))


if __name__ == "__main__":
    print("Welcome to the Handwritten Digit Recognition Project! 🖋️")

    # Step 1: Load and explore the data
    X_train, X_test, y_train, y_test = load_and_explore_data()

    # Step 2: Preprocess the data
    X_train, X_test, y_train, y_test = preprocess_data(X_train, X_test, y_train, y_test)

    # Step 3: Train the model
    print("\nTraining the Neural Network model...")
    model = train_model(X_train, y_train)

    # Step 4: Evaluate the model
    print("\nEvaluating the model...")
    evaluate_model(model, X_test, y_test)