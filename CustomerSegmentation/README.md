# Customer Segmentation using K-Means Clustering 🛍️

A Python project that groups customers into clusters based on their annual income and spending score using **K-Means Clustering**. The project uses the **Mall Customers Dataset** to perform unsupervised learning and visualize the results.

---

## Features

- **Dataset**: Uses the Mall Customers Dataset, which contains information about customers' age, gender, annual income, and spending score.
- **Model**: Applies K-Means clustering to group customers into distinct clusters.
- **Visualization**: Includes a scatter plot to visualize the clusters.
- **Scalability**: Preprocesses the data by scaling features to ensure optimal clustering performance.

---

## How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/CustomerSegmentation.git
cd CustomerSegmentation
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install numpy pandas scikit-learn matplotlib seaborn
python customer_segmentation.py
Welcome to the Customer Segmentation Project! 🛍️
First 5 rows of the dataset:
   CustomerID  Gender  Age  Annual Income (k$)  Spending Score (1-100)
0           1    Male   19                  15                      39
1           2    Male   21                  15                      81
...

Dataset Information:
Shape of the dataset: (200, 5)

Applying K-Means clustering...

Visualizing the clusters...
Technologies Used
Python : Core programming language.
Scikit-learn : For preprocessing, clustering, and evaluation.
Pandas & NumPy : For data manipulation and numerical operations.
Matplotlib & Seaborn : For visualizations.
CustomerSegmentation/
├── customer_segmentation.py  # Main script for clustering and visualization
└── README.md                 # This file