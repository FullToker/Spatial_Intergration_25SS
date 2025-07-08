var map = new maplibregl.Map({
  container: 'my-map',
  style: '/map_style/final.json',
});

map.addControl(new maplibregl.NavigationControl());

function updateMapInfo() {
  const zoom = map.getZoom();
  const center = map.getCenter();
  
  document.getElementById('zoom-level').textContent = zoom.toFixed(2);
  document.getElementById('center-coords').textContent = 
    `${center.lng.toFixed(6)}, ${center.lat.toFixed(6)}`;
}

map.on('load', updateMapInfo);
map.on('move', updateMapInfo);
map.on('zoom', updateMapInfo);

// Add click event listener to show feature information
map.on('click', function(e) {
  const features = map.queryRenderedFeatures(e.point);
  
  if (features.length > 0) {
    const feature = features[0];
    showFeatureInfo(feature, e.lngLat);
  } else {
    hideFeatureInfo();
  }
});

function showFeatureInfo(feature, lngLat) {
  let infoDiv = document.getElementById('feature-info');
  
  if (!infoDiv) {
    infoDiv = document.createElement('div');
    infoDiv.id = 'feature-info';
    document.body.appendChild(infoDiv);
  }
  
  let content = `<strong>Feature Information</strong><br>`;
  content += `<strong>Layer:</strong> ${feature.layer.id}<br>`;
  content += `<strong>Type:</strong> ${feature.geometry.type}<br>`;
  content += `<strong>Coordinates:</strong> ${lngLat.lng.toFixed(6)}, ${lngLat.lat.toFixed(6)}<br>`;
  
  if (feature.properties) {
    content += `<strong>Properties:</strong><br>`;
    for (const [key, value] of Object.entries(feature.properties)) {
      content += `&nbsp;&nbsp;${key}: ${value}<br>`;
    }
  }
  
  infoDiv.innerHTML = content;
  infoDiv.style.display = 'block';
}

function hideFeatureInfo() {
  const infoDiv = document.getElementById('feature-info');
  if (infoDiv) {
    infoDiv.style.display = 'none';
  }
}
