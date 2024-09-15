import time
import requests
import pandas as pd

from tqdm import tqdm

from treeevent.utils.coords import convert_epsg3067_to_wgs84, are_coordinates_close


def get_elevations_batch_with_retry(coordinates, max_retries=3, retry_delay=5):
    url = "https://api.open-elevation.com/api/v1/lookup"
    locations = [{"latitude": lat, "longitude": lon} for lat, lon in coordinates]
    data = {"locations": locations}

    retries = 0
    while retries < max_retries:
        try:
            response = requests.post(url, json=data)

            if response.status_code == 200:
                return response.json()["results"]
            elif response.status_code == 504:
                print(
                    f"504 Error: Retrying in {retry_delay} seconds... ({retries+1}/{max_retries})"
                )
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
                retries += 1
            else:
                print(f"Error: {response.status_code}")
                return None
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            return None
    print("Max retries exceeded. Could not fetch elevations.")
    return None


def get_elevations(df, batch_size=50):
    elevation_cache = {}
    results = []
    batch_coords = []
    row_indices = []

    for i, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing Coordinates"):
        x, y = row["x"], row["y"]

        lat, lon = convert_epsg3067_to_wgs84(x, y)
        coord = (lat, lon)

        for cached_coord in elevation_cache:
            if are_coordinates_close(coord, cached_coord):
                results.append(elevation_cache[cached_coord])
                break
        else:
            batch_coords.append(coord)
            row_indices.append(i)

            if len(batch_coords) >= batch_size:
                batch_results = get_elevations_batch_with_retry(batch_coords)

                if batch_results:
                    for j, result in enumerate(batch_results):
                        elevation_cache[batch_coords[j]] = result["elevation"]
                        results.append(result["elevation"])

                batch_coords.clear()
                row_indices.clear()

                time.sleep(2)  # Adjust the delay depending on rate limits

    if batch_coords:
        batch_results = get_elevations_batch_with_retry(batch_coords)

        if batch_results:
            for j, result in enumerate(batch_results):
                elevation_cache[batch_coords[j]] = result["elevation"]
                results.append(result["elevation"])

    df["elevation"] = pd.Series(results)
    return df
