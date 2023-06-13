import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib

# Global variables
data = None
kmeans_models = {}
kmeans_clusters = {}
kmeans_cluster_centers = {}
regression_models = {}
features = None

# Read the file path from Kaggle and return a DataFrame of recorded road accidents in Metro Manila from 2018-2020
def get_data():
    global data, features
    if data is None:
        data = pd.read_csv("data_mmda_traffic_spatial.csv")

        ## PREPROCESS
        data.dropna(subset=["Date", "Time", "Longitude", "Latitude"], inplace=True)
        data = data[(data['Latitude'] != 0.0) & (data['Longitude'] != 0.0)
                    & (data['Time'] != 0.0) & (data['Date'] != 0.0)]
        data.drop(["Source", "High_Accuracy", "Tweet", "Direction", "Type",
                "Lanes_Blocked", "Involved"], axis=1, inplace=True)
        data.drop_duplicates(inplace=True)

        # Convert Longitude and Latitude to float
        data["Longitude"] = data["Longitude"].astype(float)
        data["Latitude"] = data["Latitude"].astype(float)

        # Combines the Date and Time to datetime format and maps to their corresponding ordinal representation
        data["Datetime"] = pd.to_datetime(data["Date"])
        data["month"] = data["Datetime"].dt.month
        data["year"] = data["Datetime"].dt.year

        features = data[["Latitude", "Longitude"]]

    return data, features

def train_kmeans_models(features):
    for K in range(1, 11):
        print('Training KMeans Models for ', K, ' clusters')
        kmeans = KMeans(n_clusters=K, random_state=0)
        clusters = kmeans.fit_predict(features)
        kmeans_models[K] = kmeans
        kmeans_clusters[K] = clusters
        kmeans_cluster_centers[K] = kmeans.cluster_centers_

def train_regression_models(_data, k):
    # Group the data by cluster and month, and count the number of accidents
    grouped_data = _data.groupby(
        ["cluster", "month"]).size().reset_index(name='accidents')

    cluster_datasets = {}
    for cluster_id in range(k):
        # Finalize current cluster's data for the Month and Accident Count column
        cluster_data = grouped_data[grouped_data["cluster"] == cluster_id]
        cluster_data = cluster_data[['cluster', 'month', 'accidents']]
        cluster_datasets[cluster_id] = cluster_data

    # Store regression models based on random_state of train_test_split
    r_models = {}
    average_r2 = []
    for R in range(0, 51):
        r2_sum = 0
        # Create regression models for each cluster
        models = {}
        print('     Testing with random state ', R)
        for cluster_id in range(k):
            cluster_dataset = cluster_datasets[cluster_id]
            X_cluster = cluster_dataset[['month']]
            y_cluster = cluster_dataset['accidents']

            X_train, X_test, y_train, y_test = train_test_split(
                X_cluster, y_cluster, test_size=0.3, random_state=R)

            # Create and train the Random Forest Regression model
            model = RandomForestRegressor(random_state=0)
            model.fit(X_train, y_train)
            r2 = model.score(X_test, y_test)
            r2_sum = r2_sum + r2

            models[cluster_id] = model
        ave = r2_sum / k
        average_r2.append(ave)
        r_models[R] = models

    model_index = average_r2.index(max(average_r2))
    return r_models[model_index]

def train_all_regression_models():
    global regression_models, kmeans_clusters, kmeans_cluster_centers
    for K in range(1, 11):
        _data, _ = get_data()
        cluster_labels = kmeans_clusters[K]
        _data['cluster'] = cluster_labels
        print('Training Regression Models for ', K, ' clusters')
        r_models = train_regression_models(_data, K)
        regression_models[K] = r_models

if __name__ == '__main__':
    _, features = get_data()
    train_kmeans_models(features)
    train_all_regression_models()

    joblib.dump(kmeans_cluster_centers, 'kmeans_cluster_centers.pkl')
    joblib.dump(kmeans_clusters, 'kmeans_clusters.pkl')
    joblib.dump(features, 'features.pkl')
    joblib.dump(regression_models, 'regression_models.pkl')