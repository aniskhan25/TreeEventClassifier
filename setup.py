from setuptools import setup, find_packages

setup(
    name="TreeEvent",
    version="0.1.0",
    author="@nis",
    license="GPLv3+",
    packages=find_packages(include=["treeevent"]),
    install_requires=[
        'climatepix @ git+https://github.com/aniskhan25/climate-pix.git@aa017b462dc54975926e620e08bd1ca0676d60bd',
        'geopandas',
        'geopy',
        'numpy',
        'pandas',
        'rasterio',
        'scikit-learn',
        'shap',
        'torch',
        'torch-geometric',
        'xgboost',
    ],
)
