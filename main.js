/*
import './style.css';
import {Map, View} from 'ol';
import TileLayer from 'ol/layer/Tile';
import OSM from 'ol/source/OSM';
import 'ol/ol.css';
import { apply } from 'ol-mapbox-style';

const map = new Map({
  target: 'map',
  layers: [
    new TileLayer({
      source: new OSM()
    })
  ],
  view: new View({
    center: [0, 0],
    zoom: 2
  })
});

apply('http://localhost:8080/styles/custom-style/style.json', 'map').then(({ map }) => {
  console.log('地图加载成功');
}).catch((err) => {
  console.error('地图加载失败:', err);
});
//http://localhost:8080/styles/custom-style/style.json
*/
var map = new maplibregl.Map({
  container: 'my-map',
  style: 'map.json',
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
