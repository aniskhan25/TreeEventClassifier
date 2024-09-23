import numpy as np
import pandas as pd

from concurrent.futures import ThreadPoolExecutor

from climatepix.climate_fetcher import fetch_climate_data


def fetch_for_coords(df):
    coords = df[['x','y']].astype(int)
    year = df['year'].unique()[0]
    year = max(1950, min(year, 2022))
    
    period = f"{year}-01-01:{year}-12-31"

    return fetch_climate_data(
        coords_df=coords,
        input_crs="EPSG:3067",
        period=period,  # Use the dynamic period
        aggregation_level="Monthly",
        climatic_vars=["Prcp","Tmin","Tmax"],
    )


def fetch_climate_data_parallel(coords, years, num_workers=4):
    combined_df = pd.concat([coords, years], axis=1)

    chunk_size = 100

    chunks = []
    for year, group in combined_df.groupby('year'):
        for i in range(0, len(group), chunk_size):
            chunk = group.iloc[i:i + chunk_size]
            chunks.append(chunk)
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(fetch_for_coords, chunks))
    
    combined_df = pd.concat(results, ignore_index=True)

    invalid_value_threshold = -3.4e+38
    combined_df[['Prcp', 'Tmin', 'Tmax']] = combined_df[['Prcp', 'Tmin', 'Tmax']].map(
        lambda x: np.nan if np.isclose(x, invalid_value_threshold, atol=1e+30) else x
    )

    reorganized_df = reorganize_dataframe(combined_df)

    return reorganized_df


def reorganize_dataframe(df):
    df['day'] = pd.to_datetime(df['day'])
    df['month'] = df['day'].dt.month

    pivot = df.pivot_table(
        index=['x', 'y'],
        columns='month',
        values=['Prcp', 'Tmin', 'Tmax']
    )

    pivot.columns = [
        f'{var}_month_{month}'
        for var, month in pivot.columns
    ]

    pivot.reset_index(inplace=True)
    return pivot