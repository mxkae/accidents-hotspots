from flask import Flask, render_template, jsonify, request
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import folium
from folium.plugins import MarkerCluster

app = Flask(__name__, static_folder='static')

# Read the file path from Kaggle and return a DataFrame of recorded road accidents in Metro Manila from 2018-2020
data = pd.read_csv("data_mmda_traffic_spatial.csv")

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

# Subset of columns that will be used for clustering
features = data[["Latitude", "Longitude"]]
lat_lng = features.values

# This can be changed!
n_clusters=4

# Perform K-means with the optimal number of clusters
kmeans = KMeans(n_clusters, random_state=0)

# Assign and label each data record to a cluster
cluster_labels = kmeans.fit_predict(features)
data["cluster"] = cluster_labels


# Get the cluster centers
cluster_centers = kmeans.cluster_centers_

# Convert the "Datetime" column to datetime type
data["Datetime"] = pd.to_datetime(data["Datetime"])

# Create a Folium map centered on the mean of the cluster centers
colors =[
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


map_center = cluster_centers.mean(axis=0)
metro_coords = (14.599574, 121.059929)
m = folium.Map(
    location=map_center,
    zoom_start=11,
    tiles='OpenStreetMap'
)

# Create a MarkerCluster layer for the clusters
marker_cluster = MarkerCluster()

# Add markers to the map based on the cluster centers
for center in cluster_centers:
    folium.Marker(location=center).add_to(m)

for point, label in zip(lat_lng, cluster_labels):
    folium.CircleMarker(location=point, color=f'{colors[label]}', fill=True, radius=3).add_to(m)
    
# Display the map
m

# Group the data by cluster and month, and count the number of accidents
grouped_data = data.groupby(["cluster", "month"]).size().reset_index(name='accidents')

# Create a new DataFrame to store the accident counts for each cluster and month
cluster_datasets = {}
for cluster_id in range(n_clusters): 
    # Finalize current cluster's data for the Month and Accident Count column
    cluster_data = grouped_data[grouped_data["cluster"] == cluster_id]
    cluster_data = cluster_data[['cluster','month', 'accidents']]
    cluster_datasets[cluster_id] = cluster_data

# Create regression models for each cluster
models = {}
for cluster_id in range(n_clusters):
    cluster_dataset = cluster_datasets[cluster_id]
    X_cluster = cluster_dataset[['month']]
    y_cluster = cluster_dataset['accidents']
    
    X_train, X_test, y_train, y_test = train_test_split( X_cluster, y_cluster, test_size=0.3, random_state=2)
    
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
        
    print(f"Cluster {cluster_id}:")
    print(f"MSE: {mse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R^2: {r2}")
    print()
    
    models[cluster_id] = model
    
@app.route('/')
def home():
    return render_template('index.html', map_center=map_center, cluster_centers=cluster_centers, lat_lng=lat_lng, cluster_labels=cluster_labels)


@app.route('/predict/<int:month>', methods=['GET'])
def predict_1(month):
    input_data_1 = pd.DataFrame({
    'month': [month],
    })

    # Initialize variables
    highest_cluster = None
    highest_accidents = 0
    predictions = []

    # Iterate through all clusters and their regression models
    for cluster_id in range(n_clusters):
        model = models[cluster_id]
        # Predict the number of accidents for that month
        
        prediction = model.predict(input_data_1)
        print(f"Cluster {cluster_id} - Predicted Accidents for {month} : {prediction[0]:.2f}")

        predictions.append(prediction[0])
        # Take note of the highest accident count
        if prediction[0] > highest_accidents:
            highest_cluster = cluster_id
            highest_accidents = prediction[0]

    # Print the predicted highest accident count cluster
    print(f"\nCluster {highest_cluster} has the highest number of accidents with {int(highest_accidents)} predicted accidents for {month}.")
    #cluster_data = data[data['cluster'] == highest_cluster]
    #print(cluster_data)
    # Prepare the response
    response = {
        'predictions': predictions,
        'cluster_centers': cluster_centers.tolist(),
        'cluster_labels': cluster_labels.tolist(),
        'colors': colors,
        'lat_lngs': lat_lng.tolist()
    }

    return jsonify(response)

@app.route('/predict/<float:latitude>/<float:longitude>/<int:month>', methods=['GET'])
def predict_2(latitude, longitude, month):
    # Create a DataFrame with named columns
    input_data_2 = pd.DataFrame({
        'Latitude': [latitude],
        'Longitude': [longitude],
        'month': [month],
    })

    cluster = kmeans.predict(input_data_2[['Latitude', 'Longitude']])[0]
    model = models[cluster]

    prediction = model.predict(input_data_2[['month']])[0]

    print(f"Cluster is {cluster}. Predicted Accidents for {month}: {prediction:.2f}")

    return jsonify({'prediction': prediction[0]})

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)