import os
import torch
import logging
import argparse

import numpy as np
import pandas as pd
import torch.nn as nn

from torch_geometric.loader import DataLoader
from sklearn.utils.class_weight import compute_class_weight

from treeevent.utils.graph import create_graph, create_radius_graph, create_hybrid_graph
from treeevent.utils.hexmask import get_masks
from treeevent.data.feature import (
    get_coords,
    get_feature_data,
    merge_features,
    map_categorical_values,
    prepare_features,
    fill_missing_values,
    transform_features,
)
from treeevent.model.trainer import train
from treeevent.model.evaluator import evaluate
from treeevent.model.network.gnn import GNNClassifier
from treeevent.utils.importance import get_importance


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_with_early_stopping(
    train_loader: DataLoader,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scheduler: torch.optim.lr_scheduler,
    val_loader: DataLoader,
    patience: int = 10,
    epochs: int = 500,
) -> None:
    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(epochs):
        train_loss = train(train_loader, model, optimizer, criterion)
        val_loss, val_accuracy, val_balanced_accuracy, val_precision, val_recall, val_f1 = evaluate(val_loader, model, criterion, threshold=0.5)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), "best_model.pth")
            logger.info(f"Epoch {epoch+1}: Model improved. Saving model.")
        else:
            epochs_no_improve += 1

        if epochs_no_improve == patience:
            logger.info(f"Early stopping triggered after {epoch + 1} epochs.")
            break

        logger.info(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        logger.info(f"Val Accuracy: {val_accuracy:.4f}, Val Balanced Accuracy: {val_balanced_accuracy:.4f}")
        logger.info(f"Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1-Score: {val_f1:.4f}")
        scheduler.step()


def run(df: pd.DataFrame, coords: pd.DataFrame, feature_weights: list = None, patience: int = 10) -> None:
    train_mask, val_mask, test_mask = get_masks(coords)
    
    train_data, val_data, test_data = df[train_mask], df[val_mask], df[test_mask]
    train_coords, val_coords, test_coords = coords[train_mask], coords[val_mask], coords[test_mask]

    train_graph = create_radius_graph(train_data)
    val_graph = create_radius_graph(val_data)
    test_graph = create_radius_graph(test_data)
    
    #train_graph = create_hybrid_graph(train_data, train_coords, feature_weights=feature_weights)
    #val_graph = create_hybrid_graph(val_data, val_coords, feature_weights=feature_weights)
    #test_graph = create_hybrid_graph(test_data, test_coords, feature_weights=feature_weights)

    train_loader = DataLoader([train_graph], batch_size=8, shuffle=True)
    val_loader = DataLoader([val_graph], batch_size=8, shuffle=False)
    test_loader = DataLoader([test_graph], batch_size=8, shuffle=False)
    
    num_features = train_graph.x.shape[1]

    model = GNNClassifier(input_dim=num_features, hidden_dim=64, output_dim=1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

    class_weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=df["event_type"].values)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(class_weights[1], dtype=torch.float32))

    train_with_early_stopping(
        train_loader,
        model,
        optimizer,
        criterion,
        scheduler,
        val_loader,
        patience=patience,
    )

    test_loss, test_accuracy, test_balanced_accuracy, test_precision, test_recall, test_f1 = evaluate(test_loader, model, criterion, threshold=0.55)
    logger.info(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, Test Balanced Accuracy: {test_balanced_accuracy:.4f}")
    logger.info(f"Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1-Score: {test_f1:.4f}")


def main(data_folder: str, output_folder: str) -> None:
    filenames = {
        "area": "area",
        "climate": "climate",
        "clusters": "clusters",
        "elevation": "elevation",
        "ndvi": "ndvi",
    }

    df = get_coords(os.path.join(output_folder, "area.csv"), data_folder)
    coords_df = df[["x", "y"]]
    years = df[["year"]]

    dataframes = get_feature_data(data_folder, output_folder, filenames, coords_df, years)
    df_features = merge_features(dataframes)
    df_features = map_categorical_values(df_features)

    X, coords, y = prepare_features(df_features)

    categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_features = X.columns.difference(categorical_features)

    X = fill_missing_values(X, categorical_features)

    X_transformed = transform_features(X, categorical_features, numeric_features)

    important_features, _ = get_importance(X_transformed, y)
    df_transformed = pd.concat([X_transformed[important_features], y], axis=1)
    #df_transformed = pd.concat([X_transformed, y], axis=1)

    run(df_transformed, coords)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tree Event Classifier.")
    parser.add_argument("--data-folder", type=str, required=True, help="Path to the folder containing TIFF and Geojsons files.",)
    parser.add_argument("--output-folder", type=str, required=True, help="Path to the output folder.")

    args = parser.parse_args()

    main(args.data_folder, args.output_folder)


"""
Usage:

python -m treeevent.main \
    --data-folder /Users/anisr/Documents/dead_trees/Finland/RGBNIR/25cm \
    --output-folder ./output

"""
