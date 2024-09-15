import pyproj
import logging

from pyproj import Transformer
from geopy.distance import geodesic

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def save_dataframe_to_csv(df, output_file):
    df.to_csv(output_file, index=False)
    logging.info(f"DataFrame saved to {output_file}")


def convert_epsg3067_to_wgs84(x, y):
    transformer = Transformer.from_crs("EPSG:3067", "EPSG:4326", always_xy=True)
    return transformer.transform(x, y)


def are_coordinates_close(coord1, coord2, threshold=30):
    return geodesic(coord1, coord2).meters < threshold


def count_mismatches(dfOne, dfTwo):
    mismatches_x = dfOne['x'] != dfTwo['x']
    mismatches_y = dfOne['y'] != dfTwo['y']

    total_mismatches = mismatches_x | mismatches_y

    num_mismatches = total_mismatches.sum()

    print(f"Number of mismatches: {num_mismatches}")

    mismatched_rows = dfOne[total_mismatches]
    print(mismatched_rows)


def generate_nearby_points(coords, distance_meters):
    transformer_to_wgs84 = pyproj.Transformer.from_crs("EPSG:3067", "EPSG:4326", always_xy=True)
    
    nearby_points = {}
    
    for id, (x, y) in coords.iterrows():
        lat, lon = transformer_to_wgs84.transform(x, y)
        
        nearby_coords = []
        for dx, dy in [(distance_meters, 0), (-distance_meters, 0), (0, distance_meters), (0, -distance_meters)]:
            x_offset = x + dx
            y_offset = y + dy
            lat_offset, lon_offset = transformer_to_wgs84.transform(x_offset, y_offset)
            nearby_coords.append((lat_offset, lon_offset))
        
        nearby_points[(lat, lon)] = nearby_coords
    
    return nearby_points


def compute_slope_for_centroids(df, nearby_points, distance_meters):
    slopes = []
    
    for (lat, lon), nearby_coords in nearby_points.items():

        central_elevation = fetch_elevation_from_api([(lat, lon)])[0]
        
        nearby_elevations = fetch_elevation_from_api(nearby_coords)
        
        slope = calculate_slope(central_elevation, nearby_elevations, distance_meters)
        slopes.append(slope)
    
    df['slope'] = slopes
    return df
