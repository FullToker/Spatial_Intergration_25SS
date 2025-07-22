"""
Shapefile Reader Class

Author: Yushuo Feng, Claude
Date: 10.07.2025

"""

import geopandas as gpd
import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
import warnings
from shapely.geometry import Point, Polygon, MultiPolygon
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ShapefileReader:
    def __init__(self, encoding: str = "utf-8"):

        self.encoding = encoding
        self.data = None
        self.head = None
        self.file_path = None
        self.original_crs = None
        self.current_crs = None
        self.clean_data = None

    def read_shapefile(
        self,
        file_path: Union[str, Path],
        target_crs: Optional[str] = None,
        # bbox: Optional[tuple] = None,
        columns: Optional[List[str]] = None,
    ) -> gpd.GeoDataFrame:
        """
        读取shapefile文件
        Args:
            file_path: shapefile文件路径
            target_crs: 目标坐标系，如'EPSG:4326'
            columns: 要读取的列名列表
        Returns:
            GeoDataFrame对象
        """
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"文件不存在: {file_path}")

            logger.info(f"正在读取shapefile: {file_path}")

            read_kwargs = {"encoding": self.encoding}
            if columns:
                read_kwargs["columns"] = columns

            self.data = gpd.read_file(str(file_path), **read_kwargs)
            self.head = self.data.head()
            self.file_path = file_path
            self.original_crs = self.data.crs
            self.current_crs = self.data.crs

            logger.info(f"成功读取 {len(self.data)} 个要素")
            logger.info(f"原始坐标系: {self.original_crs}")

            if target_crs and target_crs != str(self.data.crs):
                self.transform_crs(target_crs)
            self._validate_data()

            return self.data
        except Exception as e:
            logger.error(f"读取shapefile失败: {e}")
            raise

    def transform_crs(self, target_crs: str) -> None:
        if self.data is None:
            raise ValueError("请先读取数据")

        try:
            logger.info(f"转换坐标系: {self.current_crs} -> {target_crs}")
            self.data = self.data.to_crs(target_crs)
            self.current_crs = target_crs
            logger.info("坐标系转换完成")
        except Exception as e:
            logger.error(f"坐标系转换失败: {e}")
            raise

    def _validate_data(self) -> None:
        if self.data is None:
            return
        # empty geo
        null_geom_count = self.data.geometry.isnull().sum()
        if null_geom_count > 0:
            logger.warning(f"发现 {null_geom_count} 个空几何对象")

        # not valid geo
        invalid_geom_count = (~self.data.geometry.is_valid).sum()
        if invalid_geom_count > 0:
            logger.warning(f"发现 {invalid_geom_count} 个无效几何对象")

        # Remove invalid geometries
        self.clean_data = self.data[self.data.geometry.is_valid].copy()
        logger.info(f"移除无效几何对象后，剩余 {len(self.clean_data)} 个要素")

    def split_multipolygons(self, old_id: str = "BuildingId") -> None:
        """
        Split multipolygons into individual polygons and add ori_id column
        """
        if self.data is None:
            raise ValueError("请先读取数据")
        self._validate_data()
        if self.clean_data is None:
            raise ValueError("请先验证cleaned_data")

        '''
        self.clean_data["ori_id"] = None
        new_rows = []
        rows_to_drop = []

        for idx, row in self.clean_data.iterrows():
            geom = row.geometry

            if isinstance(geom, MultiPolygon):
                original_id = row.get(old_id)

                # Split multipolygon into individual polygons
                for poly in geom.geoms:
                    new_row = row.copy()
                    new_row.geometry = poly
                    new_row["ori_id"] = original_id
                    new_rows.append(new_row)

                rows_to_drop.append(idx)

        # Remove original multipolygon rows
        if rows_to_drop:
            self.clean_data = self.clean_data.drop(rows_to_drop)

        # Add new polygon rows
        if new_rows:
            new_gdf = gpd.GeoDataFrame(new_rows, crs=self.current_crs)
            self.clean_data = pd.concat([self.clean_data, new_gdf], ignore_index=True)
        '''
        self.clean_data = self.clean_data.explode(index_parts=False)
        self.clean_data["ori_id"] = self.clean_data[old_id]

        logger.info(f"分割multipolygons后，共有 {len(self.clean_data)} 个要素")

    def add_centroid_column(self) -> None:
        """
        Add centroid column to calculate the centroid of all geometries.
        For polygons with holes, uses representative_point() to ensure 
        the point is inside the polygon interior.
        """
        if self.clean_data is None:
            raise ValueError("请先验证是否有清洗的数据")
        
        # Use representative_point() instead of centroid to ensure point is inside polygon
        self.clean_data["centroid"] = self.clean_data.geometry.representative_point()

        logger.info("已添加质心列")

    def fill_holes_and_simplify(self, tolerance: float = 0.3) -> None:
        """
        Fill holes in polygons and simplify geometries
        Args:
            tolerance: Simplification tolerance (higher = more simplified)
        """
        if self.clean_data is None:
            raise ValueError("There is no cleaned_data")

        def process_geometry(geom):
            if geom is None:
                return geom
            
            if isinstance(geom, Polygon):
                # Fill holes by using only exterior coordinates
                filled_geom = Polygon(geom.exterior.coords)
                # Simplify the geometry
                simplified_geom = filled_geom.simplify(tolerance, preserve_topology=True)
                return simplified_geom
            
            return geom

        original_count = len(self.clean_data)
        self.clean_data["geometry"] = self.clean_data["geometry"].apply(process_geometry)
        
        # Remove invalid geometries after processing
        self.clean_data = self.clean_data[self.clean_data.geometry.is_valid].copy()
        
        logger.info(f"Fill holes and simplify: {original_count} -> {len(self.clean_data)} items")

    def calculate_node_statistics(self) -> Dict[str, Any]:
        """
        Calculate node statistics for all polygons in clean_data
        Returns:
            Dictionary with total nodes, total buildings, and average nodes per building
        """
        if self.clean_data is None:
            raise ValueError("There is no cleaned data!")

        def count_nodes(geom):
            if geom is None:
                return 0
            if isinstance(geom, Polygon):
                # Count exterior nodes (subtract 1 because first and last points are the same)
                return len(geom.exterior.coords) - 1
            return 0

        node_counts = self.clean_data.geometry.apply(count_nodes)
        total_nodes = node_counts.sum()
        total_buildings = len(self.clean_data)
        avg_nodes_per_building = total_nodes / total_buildings if total_buildings > 0 else 0

        stats = {
            "Num of nodes": int(total_nodes),
            "Num of buildings": int(total_buildings),
            "Average node per building": round(avg_nodes_per_building, 2),
            "Least num of node": int(node_counts.min()) if not node_counts.empty else 0,
            "Most num of node": int(node_counts.max()) if not node_counts.empty else 0,
            "Med num of node": round(node_counts.median(), 2) if not node_counts.empty else 0
        }

        logger.info(f"Num of nodes ={stats['Num of nodes']}, Average node={stats['Average node per building']}")

        print("=" * 50)
        for key, value in stats.items():
            print(f"{key}: {value}")
        print("=" * 50)
        
        return stats

    def get_info(self) -> Dict[str, Any]:
        if self.data is None:
            return {"error": "Don't load any data"}

        info = {
            "文件路径": str(self.file_path),
            "要素数量": len(self.data),
            "列数": len(self.data.columns),
            # "Head": self.head,
            "列名": list(self.data.columns),
            "原始坐标系": str(self.original_crs),
            "当前坐标系": str(self.current_crs),
            "边界框": self.data.total_bounds.tolist(),
            "几何类型": self.data.geometry.geom_type.value_counts().to_dict(),
            "空几何数量": self.data.geometry.isnull().sum(),
            "无效几何数量": (~self.data.geometry.is_valid).sum(),
        }

        return info

    def print_info(self) -> None:
        """打印数据信息"""
        info = self.get_info()

        print("=" * 50)
        print("Shapefile info")
        print("=" * 50)

        for key, value in info.items():
            if isinstance(value, list) and len(value) > 10:
                print(f"{key}: {value[:5]}... (共{len(value)}个)")
            else:
                print(f"{key}: {value}")

        print("=" * 50)

    def filter_by_bbox(self, bbox: tuple) -> gpd.GeoDataFrame:
        """
        Args:
            bbox: Bounding Box (minx, miny, maxx, maxy)

        Returns:
            过滤后的GeoDataFrame
        """
        if self.data is None:
            raise ValueError("请先读取数据")

        minx, miny, maxx, maxy = bbox
        mask = (
            (self.data.geometry.bounds["minx"] <= maxx)
            & (self.data.geometry.bounds["maxx"] >= minx)
            & (self.data.geometry.bounds["miny"] <= maxy)
            & (self.data.geometry.bounds["maxy"] >= miny)
        )

        filtered_data = self.data[mask].copy()
        logger.info(f"边界框过滤: {len(self.data)} -> {len(filtered_data)} 个要素")

        return filtered_data

    def filter_by_polygon(self, polygon: Union[Polygon, MultiPolygon]) -> gpd.GeoDataFrame:
        """
        Filter data that are within the given polygon
        Args:
            polygon: Shapely Polygon or MultiPolygon object to filter with

        Returns:
            Filtered GeoDataFrame containing features that are within the polygon
        """
        if self.data is None:
            raise ValueError("Please Load the data firstly!")
        if not isinstance(polygon, (Polygon, MultiPolygon)):
            raise ValueError("Must be Shapely Polygon or MultiPolygon objects")

        mask = self.data.geometry.within(polygon)
        filtered_data = self.data[mask].copy()
        self.data = filtered_data
        logger.info(f"Polygon filter: {len(self.data)} -> {len(filtered_data)} items")

        return filtered_data

    def get_sample(self, n: int = 5) -> gpd.GeoDataFrame:
        if self.data is None:
            raise ValueError("请先读取数据")

        return self.data.head(n)

    def normalize_coordinates(
        self, target_crs: str, x0: float, y0: float
    ) -> gpd.GeoDataFrame:
        """
        Normalize coordinates in polygon and multipolygon geometries
        Args:
            target_crs: Target coordinate reference system
            x0: X coordinate offset for normalization (xnew = x - x0)
            y0: Y coordinate offset for normalization (ynew = y - y0)
        Returns:
            GeoDataFrame with normalized coordinates
        """
        if self.data is None:
            raise ValueError("请先读取数据")
        if self.current_crs != target_crs:
            self.transform_crs(target_crs)

        def normalize_coords(coords, x0, y0):
            """Normalize a list of coordinates"""
            return [(x - x0, y - y0) for x, y in coords]

        def normalize_geometry(geom, x0, y0):
            """Normalize geometry coordinates"""
            if geom is None:
                return geom

            if isinstance(geom, Polygon):
                exterior_coords = normalize_coords(list(geom.exterior.coords), x0, y0)
                interiors = [
                    normalize_coords(list(interior.coords), x0, y0)
                    for interior in geom.interiors
                ]
                return Polygon(exterior_coords, interiors)

            elif isinstance(geom, MultiPolygon):
                normalized_polygons = [
                    normalize_geometry(poly, x0, y0) for poly in geom.geoms
                ]
                return MultiPolygon(normalized_polygons)

            else:
                logger.warning(
                    f"Geometry type {type(geom)} not supported for normalization"
                )
                return geom

        data_copy = self.data.copy()
        if target_crs and target_crs != str(data_copy.crs):
            data_copy = data_copy.to_crs(target_crs)

        data_copy["geometry"] = data_copy["geometry"].apply(
            lambda geom: normalize_geometry(geom, x0, y0)
        )

        logger.info(f"Coordinates normalized with offset ({x0}, {y0})")

        return data_copy

    def export_to_file(
        self,
        output_path: Union[str, Path],
        driver: str = "ESRI Shapefile",
        encoding: str = "utf-8",
    ) -> None:
        """
        导出数据到文件
        Args:
            output_path: 输出文件路径
            driver: 输出格式驱动
            encoding: 编码格式
        """
        if self.data is None:
            raise ValueError("请先读取数据")

        try:
            self.data.drop(columns=["centroid"]).to_file(output_path, driver=driver, encoding=encoding)
            logger.info(f"数据已导出到: {output_path}")
        except Exception as e:
            logger.error(f"导出失败: {e}")
            raise
    
    def export_clean_to_file(
        self,
        output_path: Union[str, Path],
        driver: str = "ESRI Shapefile",
        encoding: str = "utf-8",
    ) -> None:
        """
        导出clean data到文件
        """
        if self.clean_data is None:
            raise ValueError("请先读取数据")

        try:
            self.clean_data.drop(columns=["centroid"]).to_file(output_path, driver=driver, encoding=encoding)
            logger.info(f"数据已导出到: {output_path}")
        except Exception as e:
            logger.error(f"导出失败: {e}")
            raise

    def __get_bound__(self) -> list:
        if self.data is None:

            raise ValueError("请先读取数据")
        return self.data.total_bounds.tolist()

    def __get_cleaned_data__(self):
        self.split_multipolygons()
        self.add_centroid_column()
        logging.info("cleaning done!")
        return self.clean_data


class OSMBuildingReader(ShapefileReader):
    """
    OSM建筑物数据专用读取器
    继承自ShapefileReader，添加OSM建筑物特定功能
    """

    def __init__(self, encoding: str = "utf-8"):
        super().__init__(encoding)
        self.building_types = []

    def read_osm_buildings(
        self, file_path: Union[str, Path], target_crs: str = "EPSG:3857"
    ) -> gpd.GeoDataFrame:
        """
        读取OSM建筑物数据
        Args:
            file_path: OSM建筑物shapefile路径
            target_crs: 目标坐标系
        Returns:
            OSM建筑物GeoDataFrame
        """
        self.read_shapefile(file_path, target_crs)
        self._analyze_building_types()
        return self.data

    def _analyze_building_types(self) -> None:
        if self.data is None:
            return

        building_cols = [
            col
            for col in self.data.columns
            if "building" in col.lower() or "type" in col.lower()
        ]

        if building_cols:
            for col in building_cols:
                if col in self.data.columns:
                    types = self.data[col].value_counts()
                    self.building_types.append(
                        {"column": col, "types": types.to_dict()}
                    )

    def get_building_stats(self) -> Dict[str, Any]:
        if self.data is None:
            return {"error": "未加载数据"}

        stats = self.get_info()

        stats["type stat:"] = self.building_types

        # area stat, before calculate the area, better under the suitable coords
        if not self.data.empty:
            areas = self.data.geometry.area
            stats["面积统计"] = {
                "最小面积": areas.min(),
                "最大面积": areas.max(),
                "平均面积": areas.mean(),
                "面积中位数": areas.median(),
                "总面积": areas.sum(),
            }

        return stats


def get_center_offset(bound1: list, bound2: list) -> list:
    all_bound = [
        min(bound1[0], bound2[0]),
        min(bound1[1], bound2[1]),
        max(bound1[2], bound2[2]),
        max(bound1[3], bound2[3]),
    ]

    center = [
        0.5 * (all_bound[0] + all_bound[2]),
        0.5 * (all_bound[1] + all_bound[3]),
    ]
    return center


default_crs = "EPSG:3857"

if __name__ == "__main__":
    osm_reader = OSMBuildingReader(encoding="utf-8")
    cada_reader = OSMBuildingReader(encoding="utf-8")

    osm_path = "../data/OSM_buildings.shp"
    cada_path = "../data/Cadastre_buildings.shp"

    osm_reader.read_osm_buildings(osm_path)
    cada_reader.read_osm_buildings(cada_path)
    osm_reader.print_info()
    cada_reader.print_info()
    # print(osm_reader.get_building_stats())

    # get the coords and the center
    center = get_center_offset(osm_reader.__get_bound__(), cada_reader.__get_bound__())
    print(f"center offset is {center}")

    osm_normalized_data = osm_reader.normalize_coordinates(
        target_crs=default_crs, x0=center[0], y0=center[1]
    )

    cada_reader.split_multipolygons()
    cada_reader.add_centroid_column()
    print(cada_reader.clean_data)
