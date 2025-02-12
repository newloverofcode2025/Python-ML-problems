# Stock Price Prediction - Time Series Forecasting 📈

A Python project that predicts stock prices using a time series forecasting model. The project uses historical stock price data to train and evaluate the model.

---

## Features

- **Dataset**: Uses historical stock price data downloaded from Yahoo Finance.
- **Preprocessing**: Handles missing values and creates features/target variables.
- **Model**: Trains a Linear Regression model to predict the next day's closing price.
- **Evaluation Metrics**: Evaluates the model using:
  - **Mean Absolute Error (MAE)**: Measures the average absolute difference between actual and predicted prices.
  - **Root Mean Squared Error (RMSE)**: Penalizes larger errors more heavily.
- **Visualization**: Includes a plot comparing actual vs predicted stock prices.

---

## How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/StockPricePrediction.git
cd StockPricePrediction
Welcome to the Stock Price Prediction Project! 📈

First 5 rows of the dataset:
                  Open        High         Low       Close   Adj Close    Volume
Date                                                                             
2015-01-02  106.589996  109.060005  105.809998  109.060005  102.232147  63923200
2015-01-05  107.160004  107.720001  104.620003  105.669998  100.232147  55791000
...

Dataset Information:
Shape of the dataset: (2000, 6)

Training the Linear Regression model...

Evaluating the model...

Model Evaluation:
Mean Absolute Error (MAE): 2.34
Root Mean Squared Error (RMSE): 3.12