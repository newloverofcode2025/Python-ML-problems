# Sentiment Analysis - Classify Movie Reviews 🎬

A Python project that classifies movie reviews as **positive** or **negative** using a machine learning model. The project uses the **IMDB Movie Reviews Dataset** to train and evaluate the model.

---

## Features

- **Dataset**: Uses the IMDB Movie Reviews Dataset, which contains 50,000 labeled reviews (25,000 for training and 25,000 for testing).
- **Preprocessing**: Tokenizes and vectorizes text data using TF-IDF.
- **Model**: Trains a Logistic Regression model to classify reviews.
- **Evaluation Metrics**: Evaluates the model using:
  - **Accuracy**: Measures the proportion of correct predictions.
  - **Classification Report**: Includes precision, recall, and F1-score for each class.
  - **Confusion Matrix**: Visualizes the performance of the model.

---

## How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/SentimentAnalysis.git
cd SentimentAnalysis
2. Download the Dataset
Download the IMDB Movie Reviews dataset from this link .
Extract the dataset into a folder named data/aclImdb inside the project directory.
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install numpy pandas scikit-learn matplotlib seaborn
python sentiment_analysis.py
Welcome to the Sentiment Analysis Project! 🎬

Dataset Information:
Number of samples: 50000
Target classes: ['neg', 'pos']
Shape of the dataset: (50000, 1)

Training the Logistic Regression model...

Evaluating the model...

Model Accuracy: 0.88

Classification Report:
              precision    recall  f1-score   support
    Negative       0.88      0.89      0.88      5000
    Positive       0.88      0.88      0.88      5000

    accuracy                           0.88     10000
   macro avg       0.88      0.88      0.88     10000
weighted avg       0.88      0.88      0.88     10000

Technologies Used
Python : Core programming language.
Scikit-learn : For preprocessing, training, and evaluating the model.
Pandas & NumPy : For data manipulation and numerical operations.
Matplotlib & Seaborn : For visualizations.