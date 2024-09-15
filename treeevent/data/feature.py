import os

import numpy as np
import pandas as pd

from scipy.stats import linregress

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from treeevent.data.ndvi import get_vegetation_data
from treeevent.data.cluster import find_clusters
from treeevent.data.geojson import process_geojson_folder
from treeevent.data.climate import fetch_climate_data_parallel
from treeevent.data.elevation import get_elevations
from treeevent.utils.dataframe import save_dataframe_to_csv


def get_coords(file_path, geojson_folder):
    if os.path.exists(file_path):
        print(f"[INFO] File found: {file_path}. Loading the CSV file.")
        df = pd.read_csv(file_path)

    else:
        print(f"[INFO] File not found: {file_path}. Generating new data.")
        df = process_geojson_folder(geojson_folder)
        save_dataframe_to_csv(df, file_path)

    return df


def get_feature(file_path, tiff_folder, geojson_folder, coords=None, type="area"):
    if os.path.exists(file_path):
        print(f"[INFO] File found: {file_path}. Loading the CSV file.")
        df = pd.read_csv(file_path)
    else:
        print(f"[INFO] File not found: {file_path}. Generating new data.")

        if type == "area":
            df = process_geojson_folder(geojson_folder)
        elif type == "climate":
            df = fetch_climate_data_parallel(coords, num_workers=8)
        elif type == "clusters":
            df = find_clusters(coords, eps=20, min_samples=3)
        elif type == "elevation":
            df = get_elevations(coords, batch_size=5)
        elif type == "ndvi":
            df = get_vegetation_data(tiff_folder, geojson_folder)
        else:
            pass

        save_dataframe_to_csv(df, file_path)

    return df


def get_feature_data(tiff_folder, geojson_folder, output_folder, filenames, coords):
    dataframes = {}

    for name, ftype in filenames.items():
        df = get_feature(os.path.join(output_folder, name + ".csv"), tiff_folder, geojson_folder, coords=coords, type=ftype)
        df[["x", "y"]] = df[["x", "y"]].astype(int)
        dataframes[ftype] = df

    return dataframes


def merge_features(dataframes):
    df_features = next(iter(dataframes.values()))

    for df in list(dataframes.values())[1:]:
        df_features = pd.merge(df_features, df, on=["x", "y"], how="outer")

    return df_features.dropna(subset=["area"])


def map_categorical_values(df):
    category_mapping = {"old": 2, "mature": 1, "young": 0}
    event_mapping = {"Clustered": 1, "Isolated": 0}
    df["category"] = df["category"].map(category_mapping)
    df["event_type"] = df["event_type"].map(event_mapping)
    return df


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    X = df.drop(columns=["x", "y", "event_type", "cluster"])
    
    X = calculate_climate_stats(X)
    X = add_age_interaction_terms(X)
    X = calculate_interactions(X)
    X = calculate_other_measures(X)
    X = add_ndvi_features(X)
    
    return X


def calculate_climate_stats(X):
    days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    def fahrenheit_to_celsius(temp_f):
        return (temp_f - 32) * 5 / 9

    for climate_var in ["Tmax", "Tmin"]:
        X[[f"{climate_var}_avg_month_{i+1}" for i in range(len(days_in_month))]] = (
            pd.concat(
                [
                    fahrenheit_to_celsius(
                        X[f"{climate_var}_month_{i+1}"] / days_in_month[i]
                    )
                    for i in range(len(days_in_month))
                ],
                axis=1,
            )
        )

    tmin_cols = [col for col in X.columns if col.startswith("Tmin_avg_")]
    tmax_cols = [col for col in X.columns if col.startswith("Tmax_avg_")]
    prcp_cols = [col for col in X.columns if col.startswith("Prcp_")]

    X["Prcp_total"] = X[prcp_cols].sum(axis=1)

    X["Tmin_min"], X["Tmax_min"], X["Prcp_min"] = (
        X[tmin_cols].min(axis=1),
        X[tmax_cols].min(axis=1),
        X[prcp_cols].min(axis=1),
    )
    X["Tmin_max"], X["Tmax_max"], X["Prcp_max"] = (
        X[tmin_cols].max(axis=1),
        X[tmax_cols].max(axis=1),
        X[prcp_cols].max(axis=1),
    )
    X["Tmin_mean"], X["Tmax_mean"], X["Prcp_mean"] = (
        X[tmin_cols].mean(axis=1),
        X[tmax_cols].mean(axis=1),
        X[prcp_cols].mean(axis=1),
    )
    X["Tmin_std"], X["Tmax_std"], X["Prcp_std"] = (
        X[tmin_cols].std(axis=1),
        X[tmax_cols].std(axis=1),
        X[prcp_cols].std(axis=1),
    )
    X["Tmin_var"], X["Tmax_var"], X["Prcp_var"] = (
        X[tmin_cols].var(axis=1),
        X[tmax_cols].var(axis=1),
        X[prcp_cols].var(axis=1),
    )

    X["Tmin_range"] = X["Tmin_max"] - X["Tmin_min"]
    X["Tmax_range"] = X["Tmax_max"] - X["Tmax_min"]
    X["Prcp_range"] = X["Prcp_max"] - X["Prcp_min"]

    X["annual_mean_temp"] = (X["Tmax_mean"] + X["Tmin_mean"]) / 2
    X["PT_ratio"] = X[prcp_cols].sum(axis=1) / X[tmax_cols].sum(axis=1)
    X["temp_trend"] = X.apply(temp_trend, axis=1, days_in_month=days_in_month)
    
    X['heat_stress_months'] = (X[tmax_cols] > 25).sum(axis=1)
    X['cold_stress_months'] = (X[tmin_cols] < 0).sum(axis=1)
    
    X['dry_months'] = (X[[f'Prcp_month_{i+1}' for i in range(len(days_in_month))]] < 10).sum(axis=1)

    X['wettest_month'] = X[[f'Prcp_month_{i+1}' for i in range(len(days_in_month))]].idxmax(axis=1)
    X['driest_month'] = X[[f'Prcp_month_{i+1}' for i in range(len(days_in_month))]].idxmin(axis=1)

    climate_vars = ["Prcp", "Tmax", "Tmin"]
    for climate_var in climate_vars:
        X[f'{climate_var}_winter'] = X[[f'{climate_var}_month_12', f'{climate_var}_month_1', f'{climate_var}_month_2']].sum(axis=1)
        X[f'{climate_var}_spring'] = X[[f'{climate_var}_month_3', f'{climate_var}_month_4', f'{climate_var}_month_5']].sum(axis=1)
        X[f'{climate_var}_summer'] = X[[f'{climate_var}_month_6', f'{climate_var}_month_7', f'{climate_var}_month_8']].sum(axis=1)
        X[f'{climate_var}_fall'] = X[[f'{climate_var}_month_9', f'{climate_var}_month_10', f'{climate_var}_month_11']].sum(axis=1)
        
        X[f'{climate_var}_Q1'] = X[f'{climate_var}_month_1'] + X[f'{climate_var}_month_2'] + X[f'{climate_var}_month_3']
        X[f'{climate_var}_Q2'] = X[f'{climate_var}_month_4'] + X[f'{climate_var}_month_5'] + X[f'{climate_var}_month_6']
        X[f'{climate_var}_Q3'] = X[f'{climate_var}_month_7'] + X[f'{climate_var}_month_8'] + X[f'{climate_var}_month_9']
        X[f'{climate_var}_Q4'] = X[f'{climate_var}_month_10'] + X[f'{climate_var}_month_11'] + X[f'{climate_var}_month_12']

        X[f'{climate_var}_winter_min'] = X[[f'{climate_var}_month_12', f'{climate_var}_month_1', f'{climate_var}_month_2']].min(axis=1)

    for i in range(len(days_in_month)):
        X[f'temp_range_month_{i+1}'] = X[f'Tmax_avg_month_{i+1}'] - X[f'Tmin_avg_month_{i+1}']

    for i in range(len(days_in_month)):
        X[f'temp_precip_interaction_month_{i+1}'] = X[f'Tmax_avg_month_{i+1}'] * X[f'Prcp_month_{i+1}']

    return X


def add_age_interaction_terms(X):
    for var in ["Prcp", "Tmin", "Tmax"]:
        for age_cat, label in zip([0, 1, 2], ["young", "mature", "old"]):
            X[f"{var}_{label}"] = X[f"{var}_mean"] * (X["category"] == age_cat)
    return X


def calculate_interactions(X):
    X["Area_elevation"] = X["elevation"] * X["area"]
    X["Prcp_elevation"] = X["Prcp_total"] * X["elevation"]

    X['Prcp_area'] = X['Prcp_total'] * X['area']
    X['Prcp_Tmin'] = X['Prcp_total'] * X['Tmin_mean']
    X['Prcp_Tmax'] = X['Prcp_total'] * X['Tmax_mean']

    X['Tmin_elevation'] = X['Tmin_mean'] * X['elevation']
    X['Tmin_area'] = X['Tmin_mean'] * X['area']

    X['Tmax_elevation'] = X['Tmax_mean'] * X['elevation']
    X['Tmax_area'] = X['Tmax_mean'] * X['area'] # Total Heat Exposure per Area
    
    X["temp_elevation"] = X["annual_mean_temp"] * X["elevation"]
    X['temp_area'] = X['annual_mean_temp'] * X['area']
    return X


def calculate_other_measures(X):
    X["Prcp_per_unit_area"] = X["Prcp_total"] / X["area"]
    X['Tmax_per_unit_area'] = X['Tmax_mean'] / X['area']
    
    X["area_perimeter_ratio"] = X["area"] / (2 * np.sqrt(np.pi * X["area"]))
    X['temp_area_impact'] = (X['Tmax_mean'] * X['area']) / (X['Prcp_total'] * X['area'])
    X['temperature_variability_per_area'] = (X['Tmax_mean'] - X['Tmin_mean']) / X['area'] # Temperature Variability Scaled by Area
    X['precipitation_temp_ratio'] = (X['Prcp_total'] / X['Tmax_mean']) / X['area'] # Precipitation-to-Temperature Ratio per Area

    X['elevation_cleaned'] = X['elevation'].replace(0, 0.01)
    X['elevation_temp_adjusted'] = X['Tmax_mean'] / X['elevation_cleaned'] # Elevation-Adjusted Temperature
    return X


def add_ndvi_features(X):
    # NDVI Contrast Features
    X['NDVI_diff_mean'] = X['NDVI'] - X['NDVI_mean']
    X['NDVI_diff_max'] = X['NDVI_max'] - X['NDVI']
    X['NDVI_diff_min'] = X['NDVI'] - X['NDVI_min']

    # NDVI Ratio Features
    X['NDVI_to_mean_ratio'] = X['NDVI'] / (X['NDVI_mean'] + 1e-10)
    X['NDVI_to_max_ratio'] = X['NDVI'] / (X['NDVI_max'] + 1e-10)
    X['NDVI_to_min_ratio'] = X['NDVI'] / (X['NDVI_min'] + 1e-10)

    # NDVI Variance Features
    X['NDVI_variance_to_mean'] = X['NDVI_variance'] / (X['NDVI_mean'] + 1e-10)
    X['NDVI_std'] = np.sqrt(X['NDVI_variance'])

    # NDVI Threshold Features
    X['NDVI_stressed'] = np.where(X['NDVI'] < 0.2, 1, 0)
    X['NDVI_below_mean'] = np.where(X['NDVI'] < X['NDVI_mean'], 1, 0)

    # NDVI Gradient
    buffer_radius = 15  # Assume 15 meters buffer for NDVI mean
    X['NDVI_gradient'] = (X['NDVI'] - X['NDVI_mean']) / buffer_radius

    # NDVI Range
    X['NDVI_range'] = X['NDVI_max'] - X['NDVI_min']

    # Interaction Features
    X['NDVI_area_interaction'] = X['NDVI'] * X['area']
    X['NDVI_elevation_interaction'] = X['NDVI'] * X['elevation']
    X['NDVI_precipitation_interaction'] = X['NDVI'] * X['Prcp_total']
    return X


def fill_missing_values(X, categorical_features):
    columns_to_fill = X.columns.difference(categorical_features)
    X[columns_to_fill] = X[columns_to_fill].fillna(X[columns_to_fill].mean())
    return X


def temp_trend(row, days_in_month):
    months = list(range(1, len(days_in_month) + 1))
    tmax_values = np.array([row[f"Tmax_avg_month_{i}"] for i in months])
    slope, _, _, _, _ = linregress(months, tmax_values)
    return slope


def transform_features(X, categorical_features, numeric_features):
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    return preprocessor.fit_transform(X)
