from flask import Flask, render_template, jsonify, request
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.ensemble import RandomForestRegressor

app = Flask(__name__, static_folder='static')

DEFAULT_CLUSTERS = 4
COLORS =[
    '#2f4f4f', 
    '#2e8b57', 
    '#8b0000', 
    '#808000', 
    '#00008b', 
    '#ff0000',
    '#ff8c00',
    '#7fff00',
    '#4169e1',
    '#00ffff',
    '#00bfff',
    '#0000ff',
    '#f08080',
    '#da70d6',
    '#d8bfd8',
    '#ff00ff',
    '#eee8aa',
    '#ffff54',
    '#ff1493',
    '#98fb98'
    ]


# Read the file path from Kaggle and return a DataFrame of recorded road accidents in Metro Manila from 2018-2020
data = pd.read_csv("data_mmda_traffic_spatial.csv")
# Subset of columns that will be used for clustering

kmeans_models = {}
kmeans_clusters = {}
kmeans_cluster_centers = {}

## PREPROCESS 

# Drop rows with NaN, 0.0, irrelevant, and duplicate values
data.dropna(subset=["Date", "Time", "Longitude", "Latitude"], inplace=True)
data = data[(data['Latitude'] != 0.0) & (data['Longitude'] != 0.0) & (data['Time'] != 0.0) & (data['Date'] != 0.0)]
data.drop(["Source", "High_Accuracy", "Tweet", "Direction", "Type", "Lanes_Blocked", "Involved"], axis=1, inplace=True)
data.drop_duplicates(inplace=True)

# Convert Longitude and Latitude to float
data["Longitude"] = data["Longitude"].astype(float)
data["Latitude"] = data["Latitude"].astype(float)

# Extract the Hour, Minute, and AM/PM Information from the Time Column
time_pattern = r'(\d{1,2}):(\d{2})\s*(AM|PM)'
data[["Hour", "Minute", "AM/PM"]] = data["Time"].str.extract(time_pattern, expand=True)

# Fill-in missing values of the Hour and Minute columns with 0
data["Hour"].fillna(0, inplace=True)
data["Minute"].fillna(0, inplace=True)

# Convert the Hour and Minute Columns to integer
data["Hour"] = data["Hour"].astype(int)
data["Minute"] = data["Minute"].astype(int)

# 12-hour format to 24-hour format
data.loc[data["AM/PM"] == "PM", "Hour"] += 12

# Handles invalid hour values by using modulo division
data.loc[data["Hour"] > 23, "Hour"] = data["Hour"] % 24

# Handles invalid minute values by using modulo division
data.loc[data["Minute"] > 59, "Minute"] = data["Minute"] % 60

# New Time column with the format HH:MM
data["Time"] = data["Hour"].map("{:02d}".format) + ":" + data["Minute"].map("{:02d}".format)

# Combines the Date and Time to datetime format and maps to their corresponding ordinal representation
data["Datetime"] = pd.to_datetime(data["Date"] + " " + data["Time"])
data["month"] = data["Datetime"].dt.month
data["year"] = data["Datetime"].dt.year

features = data[["Latitude", "Longitude"]]
lat_lng = features.values

for K in range(1, 11):
    kmeans = KMeans(n_clusters=K, random_state=0, n_init=10)
    clusters = kmeans.fit_predict(features)
    kmeans_models[K] = kmeans
    kmeans_clusters[K] = clusters
    kmeans_cluster_centers[K] = kmeans.cluster_centers_


def perform_regression(_data, k):
    # Group the data by cluster and month, and count the number of accidents
    grouped_data = _data.groupby(["cluster", "month", "year"]).size().reset_index(name='accidents')

    cluster_datasets = {}
    for cluster_id in range(k): 
        # Finalize current cluster's data for the Month and Accident Count column
        cluster_data = grouped_data[grouped_data["cluster"] == cluster_id]
        cluster_data = cluster_data[['cluster', 'month', 'year', 'accidents']]
        cluster_datasets[cluster_id] = cluster_data

    # Store regression models based on random_state of train_test_split
    regression_models = {}
    average_r2 = []
    for R in range(0,51):
        r2_sum = 0
        # Create regression models for each cluster
        models = {}
        for cluster_id in range(k):
            cluster_dataset = cluster_datasets[cluster_id]
            X_cluster = cluster_dataset[['month', 'year']]
            y_cluster = cluster_dataset['accidents']
            
            X_train, X_test, y_train, y_test = train_test_split( X_cluster, y_cluster, test_size=0.3, random_state=R)
            
            # Create and train the Random Forest Regression model
            model = RandomForestRegressor(random_state=0)
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)

            # Evaluate the model
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            r2_sum = r2_sum + r2
            
            models[cluster_id] = model
        ave = r2_sum / k
        average_r2.append(ave)
        regression_models[R] = models

    model_index = average_r2.index(max(average_r2))
    return regression_models[model_index]    


@app.route('/static/css/main.css')
def serve_css():
    return app.send_static_file('css/styles.css')


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict/<int:nclusters>/<int:month>/<int:year>', methods=['GET'])
def predict(nclusters, month, year):
    global data, kmeans_clusters, kmeans_cluster_centers, lat_lng
    input_data = pd.DataFrame({
    'month': [month],
    'year': [year]
    })

    _data = data.copy()
    cluster_labels = kmeans_clusters[nclusters]
    cluster_centers = kmeans_cluster_centers[nclusters]
    _data['cluster'] = cluster_labels

    r_models = perform_regression(_data, nclusters)

    predictions = []

    for cluster_id in range(nclusters):
        model = r_models[cluster_id]

        prediction = model.predict(input_data)
        print(f"Cluster {cluster_id} - Predicted Accidents for {month} {year} : {prediction[0]:.2f}")
        predictions.append(prediction[0])
    
    response = {
        'predictions': predictions,
        'cluster_centers': cluster_centers.tolist(),
        'cluster_labels': cluster_labels.tolist(),
        'colors': COLORS,
        'lat_lngs': lat_lng.tolist()
    }

    return jsonify(response)

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)