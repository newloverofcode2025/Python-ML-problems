# Boston Housing Price Prediction 🏠

A Python project that predicts housing prices in Boston using machine learning. The project includes:
1. Experimenting with multiple regression models (Linear Regression, Decision Tree, Random Forest).
2. Hyperparameter tuning using `GridSearchCV`.
3. Evaluating the model using metrics like RMSE and R².
4. Deploying the model as a web app using Flask.

---

## Features

- **Model Choice**: Experiment with different regression models (`LinearRegression`, `DecisionTreeRegressor`, `RandomForestRegressor`).
- **Hyperparameter Tuning**: Use `GridSearchCV` to find the best hyperparameters for each model.
- **Evaluation Metrics**: Display Root Mean Squared Error (RMSE) and R² Score.
- **Deployment**: Deploy the trained model as a web app using Flask for real-time predictions.

---

## How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/BostonHousingPrediction.git
cd BostonHousingPrediction
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install numpy pandas scikit-learn matplotlib seaborn flask
python train_model.py
Follow the prompts to select a model type (linear, tree, forest).
The script will save the trained model as model.pkl.
Open your browser and go to http://127.0.0.1:5000/.
Enter housing features to predict the price.

Welcome to the Boston Housing Price Prediction! 🏠
First 5 rows of the dataset:
       CRIM    ZN  INDUS  CHAS    NOX     RM   AGE     DIS  RAD    TAX  PTRATIO       B  LSTAT  PRICE
0  0.00632  18.0   2.31   0.0  0.538  6.575  65.2  4.0900  1.0  296.0     15.3  396.90   4.98   24.0
1  0.02731   0.0   7.07   0.0  0.469  6.421  78.9  4.9671  2.0  242.0     17.8  396.90   9.14   21.6
...

Best Parameters for forest: {'max_depth': 10, 'n_estimators': 100}

Model Evaluation:
Root Mean Squared Error (RMSE): 3.12
R² Score: 0.85

BostonHousingPrediction/
├── app.py                     # Flask web app
├── train_model.py             # Script for training and evaluating models
├── templates/
│   └── index.html             # HTML template for Flask
└── README.md                  # This file

Extensibility
You can extend this project by:

Adding more models (e.g., Gradient Boosting, Support Vector Regression).
Visualizing feature importance or partial dependence plots.
Building a REST API for integration with other applications.
Containerizing the app using Docker for easy deployment.