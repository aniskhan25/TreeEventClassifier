import os
import logging

import pandas as pd
import geopandas as gpd

from concurrent.futures import ThreadPoolExecutor, as_completed

from treeevent.utils.dataframe import save_dataframe_to_csv


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def process_geojson(file_path, young_threshold, mature_threshold):
    areas, centroid_coords, age_categories = [], [], []

    try:
        logging.info(f"Processing file: {file_path}")

        gdf = gpd.read_file(file_path)

        if gdf.crs is None:
            logging.warning(f"No CRS defined for {file_path}. Skipping file.")
            return [], [], []

        if gdf.crs.to_string() != "EPSG:3067":
            gdf = gdf.to_crs(epsg=3067)

        for idx, row in gdf.iterrows():
            geometry = row["geometry"]

            if geometry is None:
                continue

            if geometry.geom_type == "MultiPolygon":
                for poly in geometry.geoms:
                    area, centroid, category = classify_polygon(poly, young_threshold, mature_threshold)
                    areas.append(area)
                    centroid_coords.append((centroid.x, centroid.y))
                    age_categories.append(category)

            elif geometry.geom_type == "Polygon":
                area, centroid, category = classify_polygon(geometry, young_threshold, mature_threshold)
                areas.append(area)
                centroid_coords.append((centroid.x, centroid.y))
                age_categories.append(category)

    except Exception as e:
        logging.error(f"Error processing file {file_path}: {e}")

    return areas, centroid_coords, age_categories


def classify_polygon(polygon, young_threshold, mature_threshold):
    area = polygon.area
    centroid = polygon.centroid

    if area < young_threshold:
        category = "young"
    elif young_threshold <= area < mature_threshold:
        category = "mature"
    else:
        category = "old"

    return area, centroid, category


def process_geojson_folder(
    geojson_folder, young_threshold=4, mature_threshold=10, max_workers=4
):
    geojson_files = [
        os.path.join(geojson_folder, file)
        for file in os.listdir(geojson_folder)
        if file.endswith(".geojson")
    ]

    all_areas, all_centroid_coords, all_age_categories = [], [], []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_geojson, file, young_threshold, mature_threshold)
            for file in geojson_files
        ]

        for future in as_completed(futures):
            try:
                areas, centroid_coords, age_categories = future.result()
                all_areas.extend(areas)
                all_centroid_coords.extend(centroid_coords)
                all_age_categories.extend(age_categories)
            except Exception as e:
                logging.error(f"Error during processing: {e}")

    df = pd.DataFrame(
        {
            "x": [coord[0] for coord in all_centroid_coords],
            "y": [coord[1] for coord in all_centroid_coords],
            "area": all_areas,
            "category": all_age_categories,
        }
    )

    return df


if __name__ == "__main__":
    geojson_folder = "/Users/anisr/Documents/AerialImages/Geojsons/"
    output_file = "heatmap_data.csv"

    heatmap_data = process_geojson_folder(geojson_folder)

    save_dataframe_to_csv(heatmap_data, output_file)

    print("Combined Heatmap Data:")
    print(heatmap_data.head())
