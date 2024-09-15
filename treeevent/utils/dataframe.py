import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def save_dataframe_to_csv(df, output_file):
    df.to_csv(output_file, index=False)
    logging.info(f"DataFrame saved to {output_file}")
