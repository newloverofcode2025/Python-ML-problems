A Python project that predicts house prices using Linear Regression . The project uses the California Housing Dataset from scikit-learn to train and evaluate a machine learning model.

Features
Dataset : Uses the California Housing Dataset, which contains information about housing districts in California.
Model : Trains a Linear Regression model to predict house prices based on features like median income, house age, average rooms, etc.
Evaluation Metrics : Evaluates the model using:
Root Mean Squared Error (RMSE) : Measures the average prediction error.
R² Score : Indicates how well the model explains the variance in the data.
Visualization : Includes basic data exploration and preprocessing steps.
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install numpy pandas scikit-learn matplotlib seaborn
python house_price_prediction.py
Welcome to the House Price Prediction! 🏠
First 5 rows of the dataset:
   MedInc  HouseAge  AveRooms  AveBedrms  Population  AveOccup  Latitude  Longitude  PRICE
0  8.3252      41.0  6.984127   1.023810       322.0  2.555556     37.88    -122.23  4.526
1  8.3014      21.0  6.238137   0.971880      2401.0  2.109842     37.86    -122.22  3.585
...

Dataset Information:
Features: ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']
Shape of the dataset: (20640, 9)

Training the Linear Regression model...

Evaluating the model...

Model Evaluation:
Root Mean Squared Error (RMSE): 0.75
R² Score: 0.60
Technologies Used
Python : Core programming language.
Scikit-learn : For loading the dataset, preprocessing, training, and evaluating the model.
Pandas & NumPy : For data manipulation and numerical operations.
Matplotlib & Seaborn : For visualizations (optional, not included in this version).