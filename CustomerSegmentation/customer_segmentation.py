import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_explore_data():
    """
    Loads the Mall Customers dataset and performs basic exploration.
    :return: DataFrame containing the dataset
    """
    # Load the dataset locally
    try:
        df = pd.read_csv("Mall_Customers.csv")
    except FileNotFoundError:
        print("Error: The file 'Mall_Customers.csv' was not found in the current directory.")
        print("Please download the dataset and place it in the project folder.")
        exit(1)

    print("First 5 rows of the dataset:")
    print(df.head())

    print("\nDataset Information:")
    print(f"Shape of the dataset: {df.shape}")

    return df


def preprocess_data(df):
    """
    Preprocesses the data by selecting relevant features and scaling them.
    :param df: Original DataFrame
    :return: Scaled features
    """
    # Select relevant features (Annual Income and Spending Score)
    X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled


def apply_kmeans_clustering(X_scaled, num_clusters=5):
    """
    Applies K-Means clustering to the scaled data.
    :param X_scaled: Scaled features
    :param num_clusters: Number of clusters to form
    :return: Trained K-Means model and cluster labels
    """
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    return kmeans, labels


def visualize_clusters(X_scaled, labels):
    """
    Visualizes the clusters using a scatter plot.
    :param X_scaled: Scaled features
    :param labels: Cluster labels
    """
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X_scaled[:, 0], y=X_scaled[:, 1], hue=labels, palette='viridis', s=100, legend='full')
    plt.title('Customer Segmentation using K-Means Clustering')
    plt.xlabel('Scaled Annual Income')
    plt.ylabel('Scaled Spending Score')
    plt.show()


if __name__ == "__main__":
    print("Welcome to the Customer Segmentation Project! 🛍️")

    # Step 1: Load and explore the data
    df = load_and_explore_data()

    # Step 2: Preprocess the data
    X_scaled = preprocess_data(df)

    # Step 3: Apply K-Means clustering
    print("\nApplying K-Means clustering...")
    kmeans, labels = apply_kmeans_clustering(X_scaled, num_clusters=5)

    # Step 4: Visualize the clusters
    print("\nVisualizing the clusters...")
    visualize_clusters(X_scaled, labels)