# Pseudo Code for OSM Buildings Enrichment

## Preprocessing
```
FUNCTION preprocess_buildings(osm_gdf, cadastre_gdf):
    // Remove invalid geometries
    osm_cleaned = []
    FOR each building IN osm_gdf:
        IF building.geometry.is_valid AND not building.geometry.is_empty:
            osm_cleaned.append(building)
    
    cadastre_cleaned = []
    FOR each building IN cadastre_gdf:
        IF building.geometry.is_valid AND not building.geometry.is_empty:
            cadastre_cleaned.append(building)
    
    // Split multipolygons and fill holes
    osm_processed = []
    FOR each building IN osm_cleaned:
        IF building.geometry.geom_type == 'MultiPolygon':
            FOR each polygon IN building.geometry.geoms:
                new_building = copy(building)
                new_building.geometry = Polygon(polygon.exterior.coords)
                osm_processed.append(new_building)
        ELSE:
            building.geometry = Polygon(building.geometry.exterior.coords)
            osm_processed.append(building)
    
    cadastre_processed = []
    FOR each building IN cadastre_cleaned:
        IF building.geometry.geom_type == 'MultiPolygon':
            FOR each polygon IN building.geometry.geoms:
                new_building = copy(building)
                new_building.geometry = Polygon(polygon.exterior.coords)
                cadastre_processed.append(new_building)
        ELSE:
            building.geometry = Polygon(building.geometry.exterior.coords)
            cadastre_processed.append(building)
    
    // Coordinate normalization (reproject to common CRS if needed)
    IF osm_processed.crs != cadastre_processed.crs:
        target_crs = determine_best_crs(osm_processed, cadastre_processed)
        osm_processed = osm_processed.to_crs(target_crs)
        cadastre_processed = cadastre_processed.to_crs(target_crs)
    
    // Simplify polygons using Douglas-Peucker Algorithm
    tolerance = calculate_tolerance(osm_processed, cadastre_processed)
    FOR each building IN osm_processed:
        building.geometry = building.geometry.simplify(tolerance)
    FOR each building IN cadastre_processed:
        building.geometry = building.geometry.simplify(tolerance)
    
    RETURN osm_processed, cadastre_processed
END FUNCTION
```

## Cluster
```
FUNCTION cluster_buildings(osm_gdf, cadastre_gdf):
    // Get bounding box of all buildings
    total_bounds = union_bounds(osm_gdf.total_bounds, cadastre_gdf.total_bounds)
    
    // Rasterize to image
    image_width = calculate_optimal_width(total_bounds)
    image_height = calculate_optimal_height(total_bounds)
    
    // Create raster image with building footprints
    raster_image = create_empty_image(image_width, image_height)
    
    FOR each building IN concatenate(osm_gdf, cadastre_gdf):
        pixel_coords = world_to_pixel(building.geometry, total_bounds, image_width, image_height)
        rasterize_polygon(raster_image, pixel_coords, building_value=255)
    
    // Deploy SAM model or Watershed algorithm for clustering
    IF use_sam_model:
        sam_generator = load_sam_model(device="cpu", points_per_batch=256)
        cluster_masks = sam_generator(raster_image, points_per_batch=256)
        
        // Save clustering results
        save_pickle(cluster_masks, "cluster_results.pkl")
    ELSE:
        // Use Watershed algorithm
        distance_transform = compute_distance_transform(raster_image)
        local_maxima = find_local_maxima(distance_transform)
        markers = label_markers(local_maxima)
        cluster_masks = watershed(distance_transform, markers)
    
    // Assign cluster IDs to buildings
    cluster_assignments = {}
    cluster_id = 0
    
    FOR each mask IN cluster_masks:
        current_cluster_buildings = []
        
        FOR each building IN osm_gdf:
            building_pixels = world_to_pixel(building.geometry, total_bounds, image_width, image_height)
            IF polygon_intersects_mask(building_pixels, mask):
                current_cluster_buildings.append(("osm", building.index, building))
        
        FOR each building IN cadastre_gdf:
            building_pixels = world_to_pixel(building.geometry, total_bounds, image_width, image_height)
            IF polygon_intersects_mask(building_pixels, mask):
                current_cluster_buildings.append(("cadastre", building.index, building))
        
        // Handle multipolygons - assign same ID to all parts
        FOR building_type, building_idx, building IN current_cluster_buildings:
            IF building_idx not in cluster_assignments:
                cluster_assignments[building_idx] = {"type": building_type, "cluster_id": cluster_id, "geometry": building.geometry}
        
        cluster_id += 1
    
    RETURN cluster_assignments
END FUNCTION
```

## Relationship Table
```
FUNCTION build_relationship_table(cluster_assignments):
    relationship_table = []
    clusters = group_by_cluster(cluster_assignments)
    
    FOR each cluster:
        osm_buildings = filter_by_type(cluster, "osm")
        cadastre_buildings = filter_by_type(cluster, "cadastre")
        // Handle 0:1 case - no OSM in cluster
        IF osm_buildings is empty:
            FOR each cadastre IN cadastre_buildings:
                ADD to relationship_table: {cadastre_id, relation_type: "0:1"}
            CONTINUE
        // Search each OSM building in cluster
        FOR each osm IN osm_buildings:
            matches = []
            FOR each cadastre IN cadastre_buildings:
                iou = calculate_iou(osm, cadastre)
                forward_hausdroff, backward_hausdroff = calculate_hausdorff(osm, cadastre)
                matches.append({cadastre_id, iou, forward_hausdroff, backward_hausdroff})
            
            sort(matches by iou, descending)
            relation_type = classify_relationship(matches, osm_buildings, cadastre_buildings)
            ADD to relationship_table: {osm_id, cluster_id, relation_type, related_cadastre_ids}
    
    RETURN relationship_table

FUNCTION classify_relationship(matches, osm_buildings, cadastre_buildings):
    IF matches[0].iou < 0.1:
        RETURN "1:0"
    ELIF matches[0].iou > 0.85:
        RETURN "1:1"
    ELIF count(matches where forward_hausdroff < 0.05) > half:
        RETURN "1:m"
    ELIF matches[0].backward_hausdroff == 0:
        // Check m:1 relationship - set current cadastre as A and search all other OSM
        target_cadastre = matches[0].cadastre_id
        related_osm_buildings = []
        FOR each other_osm IN osm_buildings:
            osm_to_cadastre_distance = calculate_hausdorff(other_osm, target_cadastre)
            IF osm_to_cadastre_distance.backward_hausdroff == 0:
                related_osm_buildings.append(other_osm.id)
        
        RETURN "m:1"
    ELSE:
        DO Intersection Check
        RETURN "complex" or "n:m"
END FUNCTION
```

## Relationship Table Features
The relationship table contains the following columns:
- **osm_id**: OSM building identifier (null for 0:1 relationships)
- **cadastre_id**: Primary cadastre building identifier  
- **cluster_id**: Cluster identifier
- **relation_type**: Relationship type (0:1, 1:0, 1:1, 1:m, m:1, n:m, complex)
- **related_cadastre_ids**: Array of related cadastre building IDs (for 1:m relationships)
- **related_osm_ids**: Array of related OSM building IDs (for m:1 relationships)
- **iou**: Intersection over Union score
- **hausdorff_distance**: Forward and backward Hausdorff distances
- **confidence**: Relationship confidence score

## Relationship Rules Summary
1. **0:1** - If there is no OSM building in cluster, all cadastre buildings in cluster
2. **1:0** - If the first largest IOU value is less than 0.1
3. **1:1** - If the first largest cadastre building's IOU value is larger than 0.85
4. **1:m** - If most cadastre buildings' Hausdorff distance (to OSM building) is less than 0.05
5. **m:1** - If there is zero Hausdorff distance from OSM to cadastre, set cadastre building as A, and search all other possible OSM buildings
6. **n:m** - Check all other relationships by intersection checks
7. **Human_Handle** - All other items that don't fit the above rules

## Geometry Update
```
FUNCTION update_geometry(relation_type, osm_building, cadastre_buildings):
    SWITCH relation_type:
        CASE "0:1":  // No OSM building - create new way
            new_way.geometry = cadastre_building.geometry.to_way()
            new_way.osm_type = "way"
            new_way.height = cadastre_building.height
            RETURN new_way
            
        CASE "1:0":  // No cadastre match - no change
            osm_building.edit_info = {"status": "delete"}
            RETURN osm_building
            
        CASE "1:1":  // Copy cadastre geometry
            osm_building.geometry = cadastre_building.geometry.to_way()
            osm_building.height = cadastre_building.height
            RETURN osm_building
            
        CASE "1:m":  // Merge multiple cadastre polygons
            merged_geometry = UNION(cadastre_buildings.geometries)
            osm_building.geometry = merged_geometry.simplify().to_way()
            osm_building.edit_into = {"status": "merge", "info":"Manual handle height"}
            RETURN osm_building
            
        CASE "m:1":  // Convert to relation, keep geometries
            new_relation.geometry = UNION(osm_buildings.geometries)
            new_relation.osm_type = "relation"
            new_relation.height = cadastre_building.height
            RETURN new_relation
            
        CASE "n:m":  // Flag for manual handling
            osm_building.edit_status = "manual_handle"
            RETURN osm_building
    END SWITCH
END FUNCTION
```