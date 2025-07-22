import pickle
import random
import numpy as np
from shapely.geometry import Point
import pandas as pd
import logging
from Load.rasterize import gdf2rgb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_sam_masks(pickle_path):
    """Load SAM outputs from pickle file and extract masks"""
    with open(pickle_path, "rb") as f:
        sam_outputs = pickle.load(f)

    masks = sam_outputs["masks"]
    return masks


def load_geodataframe_with_centroids(reader):
    """Load geodataframe and calculate centroids"""

    gdf = reader.clean_data
    if gdf.clean_data is None:
        logging.error("Clean data in Geo Dataframe is None!")
        raise
    if "centroid" not in gdf.columns:
        logging.error("There is no centroid info in gdf")
        raise
    # gdf["centroid"] = gdf.geometry.centroid
    return gdf


def convert_centroids_to_pixel(gdf, masks):
    """Convert geodataframe centroids to pixel coordinates"""
    height, width = masks[0].shape
    _, transform = gdf2rgb(gdf, width=width, height=height)

    pixel_coords = []
    for centroid in gdf["centroid"]:
        # Convert world coordinates to pixel coordinates
        col, row = ~transform * (centroid.x, centroid.y)
        pixel_coords.append((int(col), int(row)))

    gdf["pixel"] = pixel_coords
    return np.array(pixel_coords)


def generalize_masks(masks):
    """generate the single mask based on the original masks"""
    height, width = masks[0].shape
    generalized = np.zeros((height, width), dtype=np.int32)

    for cluster_id, mask in enumerate(masks):
        generalized[mask] = cluster_id + 1

    return generalized


def add_cluster_column(gdf, generalized_mask):
    """Add cluster ID column to geodataframe"""
    cluster_ids = []
    height, width = generalized_mask.shape

    for pixel_coord in gdf["pixel"]:
        x, y = pixel_coord
        cluster_id = generalized_mask[y, x]  # Note: y,x for numpy indexing

        # If cluster_id is 0 (background), find closest non-zero cluster
        """
        if cluster_id == 0:
            cluster_id = find_closest_nonzero_cluster(
                generalized_mask, x, y, height, width
            )
        """

        cluster_ids.append(cluster_id)
    gdf["cluster_id"] = cluster_ids
    return gdf


def find_closest_nonzero_cluster(mask, center_x, center_y, height, width):
    """Find the closest non-zero cluster ID to the given pixel coordinates"""
    for radius in range(1, max(height, width)):
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if abs(dx) != radius and abs(dy) != radius:
                    continue

                new_x = center_x + dx
                new_y = center_y + dy

                if 0 <= new_x < width and 0 <= new_y < height:
                    cluster_id = mask[new_y, new_x]
                    if cluster_id != 0:
                        return cluster_id

    # If no non-zero cluster found, return 1 as fallback
    return 1


def stat_multipolygon_cluster_ids(clustered_gdf, display: bool = False):
    """
    Analyze multipolygon cluster IDs and group related polygons.

    Efficiently groups polygons by their ori_id and collects their cluster IDs.
    Since all polygons now have ori_id after split_multipolygons, we can use
    a single groupby operation instead of multiple nested loops.
    """
    multipolygon_stats = {}

    grouped = clustered_gdf.groupby("ori_id")

    for ori_id, group in grouped:
        if len(group) > 1:
            idxs = group.index.tolist()
            cluster_ids = set(group["cluster_id"].tolist())

            multipolygon_stats[ori_id] = {"idxs": idxs, "cluster_ids": cluster_ids}

    logging.info(f"finding {len(multipolygon_stats)} multipolygons")

    # if display:
    #    print(multipolygon_stats)
    return multipolygon_stats


def multipolygon_cluster_id(clustered_gdf, mp_stats):
    """
    assign the cluster id for multiplygon and polygons with it.
    """
    # Initialize all_ids as object dtype to store lists
    clustered_gdf["all_ids"] = clustered_gdf["cluster_id"].astype(object)

    # For each multipolygon, assign same random cluster_id and set all_ids to the multipolygon cluster_ids
    for ori_id, stats in mp_stats.items():
        idxs = stats["idxs"]
        cluster_ids = stats["cluster_ids"]

        # Randomly pick one cluster_id from the multipolygon's cluster_ids
        random_cluster_id = random.choice(list(cluster_ids))
        while random_cluster_id == 0:
            random_cluster_id = random.choice(list(cluster_ids))
        cluster_ids_list = list(cluster_ids)

        # For all indices in this multipolygon group
        for idx in idxs:
            clustered_gdf.iat[idx, clustered_gdf.columns.get_loc("cluster_id")] = (
                random_cluster_id
            )
            clustered_gdf.iat[idx, clustered_gdf.columns.get_loc("all_ids")] = (
                cluster_ids_list
            )

    return clustered_gdf


def mask2geo(gdf, masks_path: str):
    # Load SAM masks
    # masks = load_sam_masks("./Cluster/mask_res/cada_masks_1024_1024.pkl")
    masks = load_sam_masks(masks_path)
    _ = convert_centroids_to_pixel(gdf, masks)
    if "pixel" not in gdf.columns:
        logging.error("not pixel info stored in GDF")
        raise

    single_mask = generalize_masks(masks)
    logging.info(
        f"mask have been converted to single, and shape is ({len(single_mask)}, {len(single_mask[0])})"
    )

    gdf = add_cluster_column(gdf, single_mask)
    logging.info("cluster id assigning done!")
    logging.info(f"Loaded {len(masks)} masks")
    logging.info(f"Added cluster IDs to {len(gdf)} polygons")

    logging.info(
        f"Polygon cluster distribution:\n {pd.Series(single_mask.reshape(-1)).value_counts().sort_index()}"
    )

    # deal with the multipolygon
    logging.info("Dealing with Multiple Polygons ...")

    mp_stats = stat_multipolygon_cluster_ids(gdf, display=True)
    clean_clustered_gdf = multipolygon_cluster_id(gdf, mp_stats)
    logging.info("All multipolygons have been assigned with same cluster id!")

    return clean_clustered_gdf
    # return masks, gdf_with_clusters, mask_cluster_ids, centroid_mask_indices


if __name__ == "__main__":
    masks = load_sam_masks("./Cluster/mask_res/cada_masks_1024_1024.pkl")
    print(len(masks))
    print(masks[0].shape)
    # masks, gdf = main()
