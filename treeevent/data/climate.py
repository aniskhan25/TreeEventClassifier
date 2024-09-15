import pandas as pd
from concurrent.futures import ThreadPoolExecutor

from climatepix.climate_fetcher import fetch_climate_data


def fetch_for_coords(coords_df_chunk):
    return fetch_climate_data(
        coords_df=coords_df_chunk, 
        input_crs="EPSG:3067", 
        period="2017-01-01:2017-12-31", 
        aggregation_level="monthly"
    )


def fetch_climate_data_parallel(df, num_workers=4):
    chunk_size = 100
    coords_chunks = [df.iloc[i:i + chunk_size] for i in range(0, len(df), chunk_size)]
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(fetch_for_coords, coords_chunks))
    
    combined_df = pd.concat(results, ignore_index=True)
    return combined_df


def reorganize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df_pivot = df.pivot_table(index=['x', 'y'], columns='month', values='value')
    df_pivot.columns = [f"month_{int(month)}" for month in df_pivot.columns]
    df_pivot = df_pivot.reset_index()
    return df_pivot