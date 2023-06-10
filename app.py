from flask import Flask, render_template, jsonify
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

app = Flask(__name__, static_folder='static')

# Global variables
COLORS = [
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

KMEANS_CLUSTERS = joblib.load('kmeans_clusters.pkl')
KMEANS_CLUSTERS_CENTERS = joblib.load('kmeans_cluster_centers.pkl')
FEATURES = joblib.load('features.pkl')
REGRESSION_MODELS = joblib.load('regression_models.pkl')

@app.route('/static/css/main.css')
def serve_css():
    return app.send_static_file('css/styles.css')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict/<int:nclusters>/<int:month>/<int:year>', methods=['GET'])
def predict(nclusters, month, year):
    global KMEANS_CLUSTERS, KMEANS_CLUSTERS_CENTERS, FEATURES, REGRESSION_MODELS, COLORS
    
    cluster_labels = KMEANS_CLUSTERS[nclusters]
    cluster_centers = KMEANS_CLUSTERS_CENTERS[nclusters]

    input_data = pd.DataFrame({
            'month': [month],
            'year': [year]
        })
    r_models = REGRESSION_MODELS[nclusters]

    predictions = []

    for cluster_id in range(nclusters):
        model = r_models[cluster_id]

        prediction = model.predict(input_data)
        print(
            f"Cluster {cluster_id} - Predicted Accidents for {month} {year} : {prediction[0]:.2f}")
        predictions.append(prediction[0])

    response = {
        'predictions': predictions,
        'cluster_centers': cluster_centers.tolist(),
        'cluster_labels': cluster_labels.tolist(),
        'colors': COLORS,
        'lat_lngs': FEATURES.values.tolist()
    }

    return jsonify(response)


# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)
