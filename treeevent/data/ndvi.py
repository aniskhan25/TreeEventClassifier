import os
import logging
import rasterio
import numpy as np
import pandas as pd
import geopandas as gpd
from pyproj import Transformer
from scipy.ndimage import uniform_filter  # For smoothing/filtering

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def compute_ndvi(nir, red):
    nir = nir.astype(np.float32)
    red = red.astype(np.float32)
    return (nir - red) / (nir + red + 1e-10)


def compute_ndvi_statistics(ndvi_band, pixel_row, pixel_col, buffer_radius=3):
    row_start = max(0, pixel_row - buffer_radius)
    row_end = min(ndvi_band.shape[0], pixel_row + buffer_radius + 1)
    col_start = max(0, pixel_col - buffer_radius)
    col_end = min(ndvi_band.shape[1], pixel_col + buffer_radius + 1)

    buffer_ndvi = ndvi_band[row_start:row_end, col_start:col_end]

    ndvi_mean = np.mean(buffer_ndvi)
    ndvi_variance = np.var(buffer_ndvi)
    ndvi_max = np.max(buffer_ndvi)
    ndvi_min = np.min(buffer_ndvi)

    return ndvi_mean, ndvi_variance, ndvi_max, ndvi_min


def process_geojson(file_path):
    centroid_coords = []

    try:
        logging.info(f"Processing file: {file_path}")

        gdf = gpd.read_file(file_path)

        if gdf.crs is None:
            logging.warning(f"No CRS defined for {file_path}. Skipping file.")
            return [], []

        if gdf.crs.to_string() != "EPSG:3067":
            gdf = gdf.to_crs(epsg=3067)

        for idx, row in gdf.iterrows():
            geometry = row["geometry"]

            if geometry is None:
                continue

            if geometry.geom_type == "MultiPolygon":
                for poly in geometry.geoms:
                    centroid = poly.centroid
                    centroid_coords.append((centroid.x, centroid.y))

            elif geometry.geom_type == "Polygon":
                centroid = geometry.centroid
                centroid_coords.append((centroid.x, centroid.y))

    except Exception as e:
        logging.error(f"Error processing file {file_path}: {e}")

    return centroid_coords, gdf.crs


def process_tiff_geojson(tiff_file, geojson_file, buffer_radius=3):
    centroids, geojson_crs = process_geojson(geojson_file)

    with rasterio.open(tiff_file) as src:
        tiff_crs = src.crs
        transformer = Transformer.from_crs(geojson_crs, tiff_crs, always_xy=True)

        nir_band = src.read(4).astype(np.float32)  # NIR band
        red_band = src.read(3).astype(np.float32)  # Red band
        ndvi_band = compute_ndvi(nir_band, red_band)  # Compute NDVI for the whole image

        results = []
        for centroid in centroids:
            lon_tiff, lat_tiff = transformer.transform(centroid[0], centroid[1])

            pixel_row, pixel_col = src.index(lon_tiff, lat_tiff)

            if (0 <= pixel_row < nir_band.shape[0]) and (0 <= pixel_col < nir_band.shape[1]):
                ndvi_value = ndvi_band[pixel_row, pixel_col]

                ndvi_mean, ndvi_variance, ndvi_max, ndvi_min = compute_ndvi_statistics(
                    ndvi_band, pixel_row, pixel_col, buffer_radius
                )

                results.append(
                    {
                        "x": centroid[0],
                        "y": centroid[1],
                        "NDVI": ndvi_value,
                        "NDVI_mean": ndvi_mean,
                        "NDVI_variance": ndvi_variance,
                        "NDVI_max": ndvi_max,
                        "NDVI_min": ndvi_min,
                    }
                )

        return pd.DataFrame(results)


def get_vegetation_data(data_folder, buffer_radius=3):
    tiff_files = {}
    geojson_files = {}

    for root, dirs, files in os.walk(data_folder):
        for f in files:
            file_stem, file_ext = os.path.splitext(f)

            if file_ext.lower() in [".tif", ".tiff"]:
                tiff_files[file_stem] = os.path.join(root, f)

            elif file_ext.lower() == ".geojson":
                geojson_files[file_stem] = os.path.join(root, f)

    all_results = pd.DataFrame(
        columns=["x", "y", "NDVI", "NDVI_mean", "NDVI_variance", "NDVI_max", "NDVI_min"]
    )

    for filename in tiff_files.keys():
        if filename in geojson_files:
            tiff_file = tiff_files[filename]
            geojson_file = geojson_files[filename]

            df = process_tiff_geojson(tiff_file, geojson_file, buffer_radius)
            all_results = pd.concat([all_results, df], ignore_index=True)
        else:
            logging.warning(f"No matching GeoJSON for {filename}.tif")

    return all_results
