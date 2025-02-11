import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def create_synthetic_data():
    """
    Creates a synthetic dataset for loan approval prediction.
    :return: DataFrame containing the dataset
    """
    # Create a larger synthetic dataset to avoid single-class splits
    data = {
        'Income': [50000, 60000, 75000, 40000, 80000, 45000, 90000, 30000, 55000, 65000,
                   48000, 52000, 67000, 35000, 72000, 58000, 85000, 28000, 50000, 70000],
        'CreditScore': [700, 720, 680, 600, 750, 620, 780, 580, 710, 730,
                        690, 710, 740, 590, 760, 680, 790, 570, 700, 740],
        'LoanAmount': [100000, 150000, 200000, 80000, 250000, 90000, 300000, 70000, 120000, 180000,
                       95000, 110000, 190000, 75000, 220000, 100000, 280000, 65000, 105000, 170000],
        'EmploymentStatus': ['Employed', 'Employed', 'Employed', 'Unemployed', 'Employed',
                             'Unemployed', 'Employed', 'Unemployed', 'Employed', 'Employed',
                             'Unemployed', 'Employed', 'Employed', 'Unemployed', 'Employed',
                             'Unemployed', 'Employed', 'Unemployed', 'Employed', 'Employed'],
        'LoanApproved': [1, 1, 1, 0, 1, 0, 1, 0, 1, 1,
                         0, 1, 1, 0, 1, 0, 1, 0, 1, 1]  # 1 = Approved, 0 = Not Approved
    }
    df = pd.DataFrame(data)

    print("Synthetic Dataset:")
    print(df)

    return df


def preprocess_data(df):
    """
    Preprocesses the data by encoding categorical variables and scaling numerical features.
    :param df: Original DataFrame
    :return: Processed features (X) and target labels (y)
    """
    # Encode categorical variable (EmploymentStatus)
    label_encoder = LabelEncoder()
    df['EmploymentStatus'] = label_encoder.fit_transform(df['EmploymentStatus'])

    # Separate features and target
    X = df.drop('LoanApproved', axis=1)
    y = df['LoanApproved']

    # Scale numerical features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y


def train_model(X_train, y_train):
    """
    Trains a Decision Tree Classifier on the training data.
    :param X_train: Scaled training features
    :param y_train: Training labels
    :return: Trained model
    """
    model = DecisionTreeClassifier(random_state=42)
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

    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {accuracy:.2f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Explicitly specify all known labels to avoid warnings
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Approved', 'Approved'],
                yticklabels=['Not Approved', 'Approved'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()


def visualize_decision_tree(model, feature_names):
    """
    Visualizes the decision tree.
    :param model: Trained Decision Tree model
    :param feature_names: Names of the features
    """
    plt.figure(figsize=(12, 8))
    plot_tree(model, feature_names=feature_names, class_names=['Not Approved', 'Approved'], filled=True)
    plt.title('Decision Tree Visualization')
    plt.show()


if __name__ == "__main__":
    print("Welcome to the Loan Approval Prediction Project! 💼")

    # Step 1: Create synthetic data
    df = create_synthetic_data()

    # Step 2: Preprocess the data
    X, y = preprocess_data(df)

    # Step 3: Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 4: Train the model
    print("\nTraining the Decision Tree Classifier...")
    model = train_model(X_train, y_train)

    # Step 5: Evaluate the model
    print("\nEvaluating the model...")
    evaluate_model(model, X_test, y_test)

    # Step 6: Visualize the decision tree
    print("\nVisualizing the decision tree...")
    feature_names = ['Income', 'CreditScore', 'LoanAmount', 'EmploymentStatus']
    visualize_decision_tree(model, feature_names)