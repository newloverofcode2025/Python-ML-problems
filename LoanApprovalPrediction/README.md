# Loan Approval Prediction using Decision Tree 💼

A Python project that predicts loan approval using a **Decision Tree Classifier**. The project uses a synthetic dataset to train and evaluate the model.

---

## Features

- **Dataset**: Uses a synthetic dataset with features like income, credit score, loan amount, and employment status.
- **Model**: Trains a Decision Tree Classifier to predict loan approval.
- **Evaluation Metrics**: Evaluates the model using:
  - **Accuracy**: Measures the proportion of correct predictions.
  - **Classification Report**: Includes precision, recall, and F1-score for each class.
  - **Confusion Matrix**: Visualizes the performance of the model.
- **Visualization**: Includes a visualization of the decision tree.

---

## How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/LoanApprovalPrediction.git
cd LoanApprovalPrediction

---

## How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/LoanApprovalPrediction.git
cd LoanApprovalPrediction
The script will:
Create a synthetic dataset for loan approval prediction.
Preprocess the data by encoding categorical variables and scaling numerical features.
Train a Decision Tree Classifier on the training data.
Evaluate the model on the testing data and display:
Accuracy score.
Classification report (precision, recall, F1-score).
Confusion matrix (visualized using a heatmap).
Visualize the decision tree.
Welcome to the Loan Approval Prediction Project! 💼
Synthetic Dataset:
   Income  CreditScore  LoanAmount EmploymentStatus  LoanApproved
0   50000          700      100000         Employed             1
1   60000          720      150000         Employed             1
...

Training the Decision Tree Classifier...

Evaluating the model...

Model Accuracy: 1.00

Classification Report:
              precision    recall  f1-score   support
           0       1.00      1.00      1.00         1
           1       1.00      1.00      1.00         1

    accuracy                           1.00         2
   macro avg       1.00      1.00      1.00         2
weighted avg       1.00      1.00      1.00         2

Visualizing the decision tree...
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install numpy pandas scikit-learn matplotlib seaborn
python loan_approval_prediction.py
