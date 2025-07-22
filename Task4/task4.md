# Task4 -- Automater Approach to Enrich OSM Buildings
Before running,
```python
pip install -e .
# pip install -r requriements.txt
```

## Load shapefile
### Info of data

#### OSM Buildings

- 要素数量: 2043
- 列名: ['full_id', 'osm_id', 'osm_type', 'building', 'name', 'type', 'id', 'geometry']
- 原始坐标系: EPSG:3857
- 边界框: [-8416068.549262377, 695892.7497381298, -8413363.307524914, 697174.6188354888]
- 几何类型: {'Polygon': 2043}

#### Cadastre Buildings


- 要素数量: 4905
- 列名: ['BuildingId', 'height', 'geometry']
- 原始坐标系: EPSG:32618
- 边界框: [-8416068.889547948, 695887.7617808486, -8413363.225300174, 697174.1225201517]
- 几何类型: {'Polygon': 4561, 'MultiPolygon': 344}


## Cluster Cadastre Buildings?
### Convert shapefile to the RGB image
Implement in "./Load/rasterize.py", convert the geodataframe(from reader) to RGB image.

### SAM model
Use the pretrained SAM model for infernece job.
```python
# load the prrtrained model
import transformers import pipeline
# device 0 : CUDA, make sure there is enough GPU memory
generator = pipeline("mask-generation", device ="cpu", points_per_batch = 256, cache_dir = "./cache")
outputs = generator(image, points_per_batch =256)
```

Save and read the results.
```python
import pickle
with open("", "wb") as f:
    pickle.dump(outputs, f)
# read the checkpoint
with open("", "rb") as f:
    outputs = pickle.load(f)
mask = outputs["masks"]
```

### Assign the cluster id
- [] Higher the resolution of the image

# Workflow
1. Data Load and Preprocessing
2. Cluster and Build Relationship Table
3. Update the OSM Buildings

# Pseudo Code
## Preprocessing
1. Remove the invalid geometry
2. Split all multipolygons, fill the hole(only keep the exterior coords)
3. Normalize the coords(if needed)
4. Simplify the ploygon(Douglas-Peucker Algorithm)
5. Convert of the Current CRS

## Cluster
1. Rasterize to the image
2. Depoly the SAM model or use the Watershed algorithm
3. Assign the ids to both Cadastre buildings and OSM Buildings
4. Deal with the multipolygons(stat and final with the same id)

## Relationship Table
1. In each cluster, build ralation tables
2. Use IOU(or Hausfroff distance) to calculate the ratio for each OSM building with every cada building in the same cluster
3. Generate a dict of each osm, like {osm_id: "cada id":[], "iou value": [], forwarf_hausdroff:[], backward: []}, the cada id and iou is ordered
4. Build the relation table according to the dict.

### Relation table
OSM_Building : Cada_building
1. if there is any osm building in this cluster, all cada buildings in this cluster is : 0:1
2. if the first larger iou value is less than 0.1, relation is: 1:0
3. if the first larger cada building's iou value is larger than 0.85,relation is 1:1 
4. if most cadabuidlings' hasudroff distance(to osm building) is less than 0.05, relation is 1:m
5. if there is zero hausfroff disctance from osm to cada, set the cada building as A, and search all other possible osmbuildings to
build the relation: m:1
6. Check all other to the n:m relations by the intersection checks.
7. All other items taggged with "Human_Handle"

## Update the feature
### Features Addition
- [] ref:buildingnum
- [] Height
- [] source:name
- [] source:geometry
- [] source:height
- [] edit_status
- [] fixme
- [] is_passed

### Geometry Update
0:1, Polygon->way
1:0, No change
1:1, Copy the geometry of the Cadastre
1:m, Merge the polygon of Cadastre and then update the OSM
m:1, Set OSM_type to relation, don’t change the geometry
n:m, Manual Handle






