$(document).ready(function () {
  var map = L.map("map").setView([14.599574, 121.059929], 11);

  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    attribution:
      'Map data &copy; <a href="https://www.openstreetmap.org/">OpenStreetMap</a> contributors',
    maxZoom: 18,
  }).addTo(map);

  var markerClusters = L.markerClusterGroup();

  // Function to handle form submission
  $("form").on("submit", function (event) {
    event.preventDefault();
    var month = $("#monthInput").val();
    if (month) {
      // Make an AJAX GET request to the prediction endpoint
      $.ajax({
        url: "/predict/" + month,
        type: "GET",
        success: function (response) {
          var predictions = response.predictions;
          var clusterCenters = response.cluster_centers;
          var clusterLabels = response.cluster_labels;
          var colors = response.colors;
          var latLngs = response.lat_lngs;

          // Display the predictions on the page
          $("#predictions").empty();
          for (var i = 0; i < predictions.length; i++) {
            $("#predictions").append(
              "<li>Predicted accidents for Cluster " +
                i +
                ": " +
                predictions[i] +
                "</li>"
            );
          }

          // Clear previous markers
          markerClusters.clearLayers();

         // Add markers to the map based on the cluster centers
        for (var i = 0; i < clusterCenters.length; i++) {
          var center = clusterCenters[i];
          var marker = L.marker(new L.LatLng(center[0], center[1]));
          marker.bindPopup("Predicted accidents: " + predictions[i].toFixed(2)); // Add a popup with the predicted number
          markerClusters.addLayer(marker);
        }

        // Add the MarkerCluster layer to the map
        map.addLayer(markerClusters);

        // Loop through the data points and plot them with their cluster color
        for (var i = 0; i < latLngs.length; i++) {
          var point = latLngs[i];
          var label = clusterLabels[i];

          var color = colors[label];
          var circleMarker = L.circleMarker(new L.LatLng(point[0], point[1]), { color: color, fillOpacity: 1, radius: 3 });
          circleMarker.addTo(map);
        }
        },
        error: function (error) {
          console.log(error);
        },
      });
    }
  });
});
