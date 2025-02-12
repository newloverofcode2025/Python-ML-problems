# Handwritten Digit Recognition - Image Classification 🖋️

A Python project that classifies handwritten digits (0–9) using a Neural Network. The project uses the **MNIST Dataset** to train and evaluate the model.

---

## Features

- **Dataset**: Uses the MNIST Dataset, which contains 70,000 grayscale images of handwritten digits (60,000 for training and 10,000 for testing).
- **Preprocessing**: Scales and reshapes the image data for input into the Neural Network.
- **Model**: Trains a simple Neural Network to classify digits.
- **Evaluation Metrics**: Evaluates the model using:
  - **Accuracy**: Measures the proportion of correct predictions.
  - **Confusion Matrix**: Visualizes the performance of the model.
  - **Classification Report**: Includes precision, recall, and F1-score for each class.
- **Visualization**: Includes a plot of a sample image and the confusion matrix.

---

## How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/HandwrittenDigitRecognition.git
cd HandwrittenDigitRecognition
Welcome to the Handwritten Digit Recognition Project! 🖋️

Dataset Information:
Training samples: 60000
Testing samples: 10000
Image shape: (28, 28)
Number of classes: 10

Training the Neural Network model...
Epoch 1/5
1500/1500 [==============================] - 5s 3ms/step - loss: 0.2456 - accuracy: 0.9265 - val_loss: 0.1234 - val_accuracy: 0.9645
...

Evaluating the model...

Model Accuracy: 0.97

Classification Report:
              precision    recall  f1-score   support
           0       0.98      0.99      0.98       980
           1       0.99      0.99      0.99      1135
           ...