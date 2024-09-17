import torch
import numpy as np

from torch_geometric.data import Data
from torch_geometric.utils import from_scipy_sparse_matrix

from sklearn.neighbors import kneighbors_graph, radius_neighbors_graph

from scipy.sparse import csr_matrix
from scipy.spatial.distance import cdist


def create_graph(data, k=5):
    feature_columns = data.columns.difference(['event_type'])

    x = torch.tensor(data[feature_columns].values.astype(np.float32), dtype=torch.float)
    y = torch.tensor(data['event_type'].values.astype(np.float32), dtype=torch.long)

    adj_matrix = kneighbors_graph(x, k, mode='connectivity', include_self=True)
    edge_index, _ = from_scipy_sparse_matrix(adj_matrix)

    return Data(x=x, edge_index=edge_index, y=y)


def create_radius_graph(data, radius=0.1):
    feature_columns = data.columns.difference(['event_type'])

    x = torch.tensor(data[feature_columns].values.astype(np.float32), dtype=torch.float)
    y = torch.tensor(data['event_type'].values.astype(np.float32), dtype=torch.long)

    adj_matrix = radius_neighbors_graph(x, radius, mode='connectivity', include_self=False)

    adj_matrix = csr_matrix(adj_matrix)  # Ensure it's in sparse format
    edge_index = torch.tensor(np.vstack((adj_matrix.nonzero())), dtype=torch.long)

    return Data(x=x, edge_index=edge_index, y=y)


def create_hybrid_graph(data, coords, spatial_k=5, similarity_threshold=0.1, feature_weights=None):
    spatial_columns = ['x', 'y']  # Assuming x, y are the spatial coordinates
    env_columns = data.columns.difference(['event_type', 'x', 'y'])  # Environmental features
    
    spatial_features = torch.tensor(coords[spatial_columns].values.astype(np.float32), dtype=torch.float)
    env_features = torch.tensor(data[env_columns].values.astype(np.float32), dtype=torch.float)

    y = torch.tensor(data['event_type'].values.astype(np.float32), dtype=torch.long)
    
    if feature_weights:
        for idx, col in enumerate(env_columns):
            env_features[:, idx] *= feature_weights[col]
    
    spatial_adj_matrix = kneighbors_graph(spatial_features, spatial_k, mode='connectivity', include_self=True)
    
    env_distances = cdist(env_features, env_features, metric='euclidean')
    env_adj_matrix = (env_distances < similarity_threshold).astype(int)
    
    combined_adj_matrix = csr_matrix(spatial_adj_matrix + env_adj_matrix)
    
    edge_index, _ = from_scipy_sparse_matrix(combined_adj_matrix)
    
    x = torch.cat([spatial_features, env_features], dim=1)
    
    return Data(x=x, edge_index=edge_index, y=y)