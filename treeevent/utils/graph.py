import torch

import numpy as np

from scipy.sparse import csr_matrix
from sklearn.neighbors import kneighbors_graph, radius_neighbors_graph

from torch_geometric.data import Data
from torch_geometric.utils import from_scipy_sparse_matrix


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