A Python project that classifies iris flowers into their respective species using machine learning. The project includes:

Experimenting with multiple models (Logistic Regression, Decision Tree, Random Forest, K-Nearest Neighbors).
Hyperparameter tuning using GridSearchCV.
Visualizing feature importance or decision boundaries.
Deploying the model as a web app using Flask.
Features
Model Choice : Experiment with different classifiers (LogisticRegression, DecisionTree, RandomForest, KNeighborsClassifier).
Hyperparameter Tuning : Use GridSearchCV to find the best hyperparameters for each model.
Evaluation Metrics : Display accuracy, classification report, and confusion matrix.
Visualization : Plot confusion matrices and decision boundaries.
Deployment : Deploy the trained model as a web app using Flask for real-time predictions.
git clone https://github.com/yourusername/IrisClassificationAdvanced.git
cd IrisClassificationAdvanced
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install numpy pandas scikit-learn matplotlib seaborn flask
Follow the prompts to select a model type (logistic, tree, forest, knn).
The script will save the trained model as model.pkl.
Open your browser and go to http://127.0.0.1:5000/.
Enter flower measurements (sepal length, sepal width, petal length, petal width) to predict the species.
