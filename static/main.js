$(document).ready(function () {
  var map = L.map("map").setView([14.599574, 121.059929], 12);

  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    attribution:
      'Map data &copy; <a href="https://www.openstreetmap.org/">OpenStreetMap</a> contributors',
    maxZoom: 18,
  }).addTo(map);

  var markerClusters = L.markerClusterGroup();

  const select_month = $('#month');
  const input_cluster = $('#clusters');
  const loadingSpinner = $('#loading-spinner');
  const map_div = $('#map');
  const note = $('#note')

  select_month.on('change', getMarkers);
  input_cluster.on('change', getMarkers);

  function getMarkers() {
    const clusters = input_cluster.val();
    const month = select_month.val();

    if (clusters && month) {

      map_div.hide();
      note.hide();
      loadingSpinner.show();

      $.ajax({
        url: "/predict/" + clusters + "/" + month,
        type: "GET",
        success: function (response) {
          const predictions = response.predictions;
          const clusterCenters = response.cluster_centers;
          const clusterLabels = response.cluster_labels;
          const colors = response.colors;
          const latLngs = response.lat_lngs;

          // Clear previous markers
          markerClusters.clearLayers();

          // Add markers to the map based on the cluster centers
          for (var i = 0; i < clusterCenters.length; i++) {
            var center = clusterCenters[i];
            var marker = L.marker(new L.LatLng(center[0], center[1]));
            marker.bindPopup("Predicted accidents: " + predictions[i].toFixed(2)).openPopup(); // Add a popup with the predicted number
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
      }).done(function() {
          loadingSpinner.hide();
          map_div.show();
          note.show();
      });

      
    }
  }
  getMarkers()
});
