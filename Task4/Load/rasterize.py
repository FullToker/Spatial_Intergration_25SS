import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_bounds
import matplotlib.pyplot as plt
from PIL import Image
from Load.shp_reader import ShapefileReader, OSMBuildingReader


def gdf2rgb(gdf, width=1024, height=1024, polygon_color=(0, 0, 255)):
    """
    Convert GeoDataFrame polygons to RGB image
    Args:
        gdf: GeoDataFrame with polygon geometries
        width: Output image width
        height: Output image height
        polygon_color: RGB color tuple for polygons (default blue)
    Returns:
        RGB numpy array (height, width, 3)
    """
    bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]

    # Create transform for rasterization
    transform = from_bounds(bounds[0], bounds[1], bounds[2], bounds[3], width, height)

    # Rasterize polygons (creates binary mask)
    shapes = [(geom, 1) for geom in gdf.geometry if geom is not None]
    mask = rasterize(
        shapes, out_shape=(height, width), transform=transform, fill=0, dtype=np.uint8
    )

    rgb_image = np.zeros((height, width, 3), dtype=np.uint8)
    polygon_pixels = mask == 1
    rgb_image[polygon_pixels] = polygon_color

    return rgb_image, transform


if __name__ == "__main__":

    reader = ShapefileReader()
    gdf = reader.read_shapefile("../data/Cadastre_buildings.shp")
    rgb_image, _ = gdf2rgb(gdf, width=2048, height=1024, polygon_color=(255, 255, 255))

    Image.fromarray(rgb_image).save("polygons_2048_1024.png")
