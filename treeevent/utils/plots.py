import os
import rasterio

import geopandas as gpd
import contextily as ctx
import matplotlib.pyplot as plt

from shapely.geometry import box


def plot_clusters(heatmap_data):
    plt.figure(figsize=(10, 8))
    
    plt.scatter(
        heatmap_data[heatmap_data['event_type'] == 'Isolated']['x'],
        heatmap_data[heatmap_data['event_type'] == 'Isolated']['y'],
        c='red', label='Isolated', alpha=0.3, s=50, marker='x'
    )
    
    plt.scatter(
        heatmap_data[heatmap_data['event_type'] == 'Clustered']['x'],
        heatmap_data[heatmap_data['event_type'] == 'Clustered']['y'],
        c='blue', label='Clustered', alpha=0.3, s=50, marker='.'
    )
    
    plt.xlabel('X Coordinate (meters)')
    plt.ylabel('Y Coordinate (meters)')
    plt.title('Spatial Distribution of Isolated and Clustered Tree Mortality Events')
    plt.legend()
    plt.show()


def plot_bounds_on_map(tiff_folder):
    bounding_boxes = []

    for file_name in os.listdir(tiff_folder):
        if file_name.endswith('.tif') or file_name.endswith('.tiff'):
            file_path = os.path.join(tiff_folder, file_name)
            
            try:
                with rasterio.open(file_path) as src:
                    bounds = src.bounds  # Get the bounding box of the image
                    crs = src.crs  # Get the CRS of the image
                    
                    gdf = gpd.GeoDataFrame({'geometry': [box(bounds.left, bounds.bottom, bounds.right, bounds.top)]}, crs=crs)
                    gdf_wgs84 = gdf.to_crs(epsg=4326)  # Convert to WGS84
                    
                    bounding_boxes.append(gdf_wgs84.geometry.iloc[0])
            except Exception as e:
                print(f"Error reading {file_name}: {e}")

    all_bounds_gdf = gpd.GeoDataFrame(geometry=bounding_boxes, crs='EPSG:4326')

    fig, ax = plt.subplots(figsize=(12, 15))
    all_bounds_gdf.plot(ax=ax, edgecolor='red', linewidth=2, marker='x', markersize=10)

    ctx.add_basemap(ax, crs=all_bounds_gdf.crs.to_string(), source=ctx.providers.OpenStreetMap.Mapnik)

    ax.set_xlim(all_bounds_gdf.total_bounds[[0, 2]])
    ax.set_ylim(all_bounds_gdf.total_bounds[[1, 3]])

    plt.title("Spatial Representation of TIFF Image Bounds Across Finland")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.show()