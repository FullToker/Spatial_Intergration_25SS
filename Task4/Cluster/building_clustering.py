import geopandas as gpd
import pandas as pd
import numpy as np
from sklearn.cluster import HDBSCAN
from sklearn.preprocessing import StandardScaler
from shapely.geometry import Point, Polygon, MultiPolygon
import matplotlib.pyplot as plt
from typing import List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def decompose_multipolygons(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Decompose MultiPolygons into individual Polygons
    Each polygon gets its own row but keeps track of original MultiPolygon
    Args:
        gdf: Original GeoDataFrame with Polygons and MultiPolygons
    Returns:
        Expanded GeoDataFrame with only Polygons
    """
    decomposed_rows = []
    multipolygon_cluster_id = 1000  # Start MultiPolygon clusters from 1000

    for idx, row in gdf.iterrows():
        if isinstance(row.geometry, MultiPolygon):
            # Decompose MultiPolygon into individual polygons
            logger.info(
                f"Decomposing MultiPolygon at index {idx} with {len(row.geometry.geoms)} parts"
            )

            for part_idx, polygon in enumerate(row.geometry.geoms):
                new_row = row.copy()
                new_row["geometry"] = polygon
                new_row["original_index"] = idx
                new_row["multipolygon_cluster"] = multipolygon_cluster_id
                new_row["part_index"] = part_idx
                new_row["is_from_multipolygon"] = True
                decomposed_rows.append(new_row)

            multipolygon_cluster_id += 1

        elif isinstance(row.geometry, Polygon):
            # Keep regular polygons as they are
            new_row = row.copy()
            new_row["original_index"] = idx
            new_row["multipolygon_cluster"] = -1  # Not from multipolygon
            new_row["part_index"] = 0
            new_row["is_from_multipolygon"] = False
            decomposed_rows.append(new_row)
        else:
            logger.warning(
                f"Skipping geometry type {type(row.geometry)} at index {idx}"
            )

    # Create new GeoDataFrame
    decomposed_gdf = gpd.GeoDataFrame(decomposed_rows, crs=gdf.crs)

    logger.info(
        f"Decomposed {len(gdf)} original geometries into {len(decomposed_gdf)} polygons"
    )
    return decomposed_gdf


def extract_centroids(gdf: gpd.GeoDataFrame) -> np.ndarray:
    """
    Extract centroids from polygon geometries
    Args:
        gdf: GeoDataFrame with Polygon geometries only
    Returns:
        Array of centroid coordinates [x, y]
    """
    centroids = gdf.geometry.centroid
    return np.column_stack([centroids.x.values, centroids.y.values])


def cluster_buildings(
    gdf: gpd.GeoDataFrame,
    min_cluster_size: int = 5,
    min_samples: int = 3,
    distance_threshold: float = 50,
) -> gpd.GeoDataFrame:
    """
    Corrected clustering with proper MultiPolygon handling
    Args:
        gdf: Building GeoDataFrame (Polygons and MultiPolygons)
        min_cluster_size: Minimum cluster size for HDBSCAN
        min_samples: Minimum samples for core points
        distance_threshold: Distance for joining MultiPolygon clusters
    Returns:
        GeoDataFrame with cluster labels
    """
    logger.info(f"Starting clustering of {len(gdf)} buildings")

    # Step 1: Decompose MultiPolygons into individual polygons
    decomposed_gdf = decompose_multipolygons(gdf)

    # Step 2: Initialize cluster IDs
    decomposed_gdf["cluster_id"] = -1

    # Step 3: Assign MultiPolygon clusters first
    multipolygon_mask = decomposed_gdf["is_from_multipolygon"] == True
    for multipolygon_cluster in decomposed_gdf[multipolygon_mask][
        "multipolygon_cluster"
    ].unique():
        mask = decomposed_gdf["multipolygon_cluster"] == multipolygon_cluster
        decomposed_gdf.loc[mask, "cluster_id"] = multipolygon_cluster
        cluster_size = mask.sum()
        logger.info(
            f"MultiPolygon cluster {multipolygon_cluster}: {cluster_size} polygons"
        )

    # Step 4: Get polygons that need clustering (not from MultiPolygons)
    unclustered_mask = decomposed_gdf["cluster_id"] == -1
    unclustered_gdf = decomposed_gdf[unclustered_mask]

    if len(unclustered_gdf) == 0:
        logger.info(
            "All polygons are from MultiPolygons, no additional clustering needed"
        )
        return decomposed_gdf

    logger.info(f"Clustering {len(unclustered_gdf)} individual polygons")

    # Step 5: Extract centroids for clustering
    centroids = extract_centroids(unclustered_gdf)
    # Step 6: Standardize coordinates
    scaler = StandardScaler()
    centroids_scaled = scaler.fit_transform(centroids)

    # Step 7: Perform HDBSCAN clustering
    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size, min_samples=min_samples, metric="euclidean"
    )

    cluster_labels = clusterer.fit_predict(centroids_scaled)

    # Step 8: Assign cluster labels to unclustered polygons
    unclustered_indices = unclustered_gdf.index
    for i, idx in enumerate(unclustered_indices):
        if cluster_labels[i] != -1:  # Not noise
            decomposed_gdf.loc[idx, "cluster_id"] = cluster_labels[i]
        # Noise points remain as -1

    # Step 9: Allow individual polygons to join MultiPolygon clusters
    decomposed_gdf = merge_nearby_to_multipolygons(decomposed_gdf, distance_threshold)

    return decomposed_gdf


def merge_nearby_to_multipolygons(
    decomposed_gdf: gpd.GeoDataFrame, distance_threshold: float = 50
) -> gpd.GeoDataFrame:
    """
    Allow individual polygons to join MultiPolygon clusters if nearby
    Args:
        decomposed_gdf: GeoDataFrame with decomposed polygons
        distance_threshold: Distance threshold for joining
    Returns:
        Updated GeoDataFrame with merged clusters
    """
    # Get MultiPolygon clusters
    multipolygon_clusters = decomposed_gdf[
        decomposed_gdf["is_from_multipolygon"] == True
    ]["multipolygon_cluster"].unique()

    if len(multipolygon_clusters) == 0:
        return decomposed_gdf
    # Get unclustered polygons (noise points and regular clusters)
    candidates = decomposed_gdf[
        (decomposed_gdf["cluster_id"] == -1)  # Noise points
        | (
            (decomposed_gdf["cluster_id"] < 1000) & (decomposed_gdf["cluster_id"] != -1)
        )  # Regular clusters
    ]

    logger.info(
        f"Checking {len(candidates)} polygons for merging with MultiPolygon clusters"
    )

    for mp_cluster_id in multipolygon_clusters:
        # Get all polygons in this MultiPolygon cluster
        mp_polygons = decomposed_gdf[
            decomposed_gdf["multipolygon_cluster"] == mp_cluster_id
        ]

        for cand_idx, cand_row in candidates.iterrows():
            cand_geometry = cand_row.geometry

            # Check distance to any polygon in the MultiPolygon cluster
            min_distance = float("inf")
            for _, mp_row in mp_polygons.iterrows():
                distance = mp_row.geometry.distance(cand_geometry)
                min_distance = min(min_distance, distance)

            if min_distance <= distance_threshold:
                old_cluster = decomposed_gdf.loc[cand_idx, "cluster_id"]
                decomposed_gdf.loc[cand_idx, "cluster_id"] = mp_cluster_id
                logger.info(
                    f"Polygon {cand_idx} moved from cluster {old_cluster} to MultiPolygon cluster {mp_cluster_id} (distance: {min_distance:.2f})"
                )

    return decomposed_gdf


def print_cluster_summary(decomposed_gdf: gpd.GeoDataFrame):
    """Print clustering summary"""
    cluster_counts = decomposed_gdf["cluster_id"].value_counts().sort_index()
    n_clusters = len(cluster_counts[cluster_counts.index != -1])
    n_noise = cluster_counts.get(-1, 0)
    n_total = len(decomposed_gdf)

    print("\n" + "=" * 60)
    print("CLUSTERING RESULTS SUMMARY")
    print("=" * 60)
    print(f"Total polygons (after decomposition): {n_total}")
    print(f"Number of clusters: {n_clusters}")
    print(f"Noise points: {n_noise} ({n_noise/n_total*100:.1f}%)")

    # Show MultiPolygon clusters
    multipolygon_clusters = decomposed_gdf[
        decomposed_gdf["is_from_multipolygon"] == True
    ]["multipolygon_cluster"].unique()
    if len(multipolygon_clusters) > 0:
        print(f"\nMultiPolygon-based clusters: {len(multipolygon_clusters)}")
        for mp_cluster_id in sorted(multipolygon_clusters):
            cluster_size = len(
                decomposed_gdf[decomposed_gdf["cluster_id"] == mp_cluster_id]
            )
            original_parts = len(
                decomposed_gdf[
                    (decomposed_gdf["multipolygon_cluster"] == mp_cluster_id)
                    & (decomposed_gdf["is_from_multipolygon"] == True)
                ]
            )
            joined_parts = cluster_size - original_parts
            print(
                f"  Cluster {mp_cluster_id}: {cluster_size} polygons ({original_parts} original + {joined_parts} joined)"
            )
    # Show regular clusters
    regular_clusters = cluster_counts[
        (cluster_counts.index != -1) & (cluster_counts.index < 1000)
    ]
    if len(regular_clusters) > 0:
        print(f"\nTop 5 regular clusters:")
        for cluster_id in regular_clusters.head().index:
            count = regular_clusters[cluster_id]
            print(f"  Cluster {cluster_id}: {count} polygons")

    print("=" * 60)


# =============================================================================
# MAIN FUNCTION FOR TESTING
# =============================================================================


def main():
    """Main function for testing corrected clustering"""
    print("Corrected Building Clustering Test")
    print("=" * 50)

    # =============================================================================
    # STEP 1: READ DATA
    # =============================================================================
    try:
        # Replace with your actual file path
        file_path = "Cadastre_buildings.shp"
        gdf = gpd.read_file(file_path)
        print(f"Successfully loaded {len(gdf)} buildings")

    except Exception as e:
        print(f"Error reading file: {e}")
        print("Creating sample data for demonstration...")
        gdf = create_sample_data()

    print(f"Original geometry types: {gdf.geometry.geom_type.value_counts().to_dict()}")

    # =============================================================================
    # STEP 2: PERFORM CORRECTED CLUSTERING
    # =============================================================================

    print("\nStarting corrected clustering...")
    clustered_gdf = cluster_buildings(
        gdf,
        min_cluster_size=3,  # Minimum 3 polygons per cluster
        min_samples=2,  # Core points need 2 neighbors
        distance_threshold=50,  # Distance for joining MultiPolygon clusters
    )

    # =============================================================================
    # STEP 3: ANALYZE RESULTS
    # =============================================================================

    print_cluster_summary(clustered_gdf)

    # =============================================================================
    # STEP 4: VISUALIZE RESULTS
    # =============================================================================

    print("\nGenerating visualization...")
    fig = visualize_clusters_corrected(clustered_gdf)
    plt.show()

    return clustered_gdf


if __name__ == "__main__":
    result = main()
