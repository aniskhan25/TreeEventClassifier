# Dead Tree Event Classifier

This program is a binary classifier designed to categorize dead tree locations into two classes: **clustered** and **isolated** events. It leverages aerial images, geographic data, and various environmental features to identify and classify dead tree events, supporting ecological analysis and forest management.

## Features

- **Geospatial Data Processing**: Integrates TIFF images and GeoJSON files to extract and process spatial information.
- **Feature Engineering**: Includes calculating climate statistics, vegetation indices (NDVI), and other interaction terms to enrich the data for classification.
- **Binary Classification**: Categorizes dead tree events into `clustered` or `isolated` categories using a combination of spatial and environmental features.
- **Command Line Interface**: Allows for flexible input of paths to data folders via command-line arguments.

## Requirements

- Python 3.x
- Required Python libraries:
  - `torch`
  - `numpy`
  - `pandas`
  - `argparse`
  - `torch_geometric`
  - `scikit_learn`
  - `rasterio`
  - `geopandas`
  - `geopy`
  - `shap`
  - `xgboost`
  - `climatepix` (pip install climatepix@git+https://github.com/aniskhan25/climate-pix.git)

  - (Other libraries needed by the internal functions, e.g., `contextily`, etc.)

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/aniskhan25/TreeEventClassifier.git
    cd TreeEventClassifier
    ```

2. Install required dependencies (if using `requirements.txt`):
    ```bash
    pip install -r requirements.txt
    ```

3. Ensure you have the following data:
   - **TIFF images** (e.g., aerial or satellite images of the area)
   - **GeoJSON files** (containing geographical data for dead tree locations)

## Usage

The classifier accepts three main folder paths as input: a folder containing TIFF files, a folder containing GeoJSON files, and an output folder for results.

### Command-Line Arguments

- `--tiff-folder`: Path to the folder containing the TIFF files.
- `--geojson-folder`: Path to the folder containing the GeoJSON files.
- `--output-folder`: Path to the output folder where results will be saved.

### Example Command

```bash
python -m treeevent.main \
    --tiff-folder /path/to/tiff_folder/ \
    --geojson-folder /path/to/geojson_folder/ \
    --output-folder /path/to/output_folder/
```

### Steps in Classification

1. **Coordinate Extraction**: Extracts coordinates from the GeoJSON files and matches them to corresponding TIFF images.
2. **Feature Data Aggregation**: Gathers environmental and spatial features (e.g., NDVI, elevation, climate stats) to construct a feature set.
3. **Feature Engineering**: Processes the features to generate additional interaction terms and statistical measures.
4. **Classification**: Uses the processed feature data to classify dead tree events into `clustered` or `isolated` categories.
5. **Output**: The results are saved to the specified output folder.

## Output

The program generates the following output:
- A transformed feature dataset with the most important features.
- A CSV file containing classified dead tree events (`clustered` or `isolated`).
- Any additional logs or visualizations (depending on your internal implementations).

## Customization

You can modify the classification logic, feature engineering steps, and the types of features used by editing the core functions in the script (e.g., `get_feature_data`, `run`, `trainer`, `evaluator`, etc.).

## Contributing

If you would like to contribute, please fork the repository and create a pull request with your changes. All contributions are welcome!

## License

This project is licensed under the GPLv3 License - see the [LICENSE](LICENSE) file for details.

## Contact

For any questions or feedback, please contact [Your Name] at [aniskhan25@gmail.com].
