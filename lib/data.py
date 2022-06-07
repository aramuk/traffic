import os
from typing import Tuple, Optional

import contextily as cx
import geopandas
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
import torch
from torch.utils.data import Dataset


class PEMSBay(Dataset):
    split_sizes: Tuple[float] = (0.70, 0.15, 0.15)

    def __init__(self, data_dir: str, split: str, hist_window: int, pred_window: int, threshold: Optional[float] = 0.1):
        if split not in ('train', 'val', 'test'):
            raise ValueError(f"Split must be one of 'train','val','test'. Received '{split}'")
        self.data_dir = data_dir
        dist_file = os.path.join(data_dir, 'distances.csv')
        sensors_file = os.path.join(data_dir, 'sensor_locations.csv')
        traffic_file = os.path.join(data_dir, 'traffic.h5')
        data_file = os.path.join(self.data_dir, f"{split}.h5")
        if not all(os.path.exists(f) for f in (dist_file, sensors_file, traffic_file)):
            raise FileNotFoundError(f"Missing dataset files: {dist_file}, {sensors_file}, or {traffic_file}")

        self.dist_df = pd.read_csv(dist_file, names=['from', 'to', 'distance'])
        self.sensors_df = pd.read_csv(sensors_file, names=['sensor_id', 'latitude', 'longitude'])
        self.adj = self._get_adj_mx(self.sensors_df.sensor_id.unique(), self.dist_df, threshold)
        self.num_vertices = self.adj.shape[0]
        self.edge_idx, self.edge_wt = self._get_edges(self.adj)

        if not os.path.exists(data_file):
            self._generate_splits(traffic_file)
        train_set = pd.read_hdf(data_file).values
        self._create_windows(train_set, hist_window, pred_window)

    def display(self, idx: int = None):
        sensors_gdf = geopandas.GeoDataFrame(
            self.sensors_df,
            geometry=geopandas.points_from_xy(self.sensors_df.longitude, self.sensors_df.latitude, crs="EPSG:4326")
        )
        ax = sensors_gdf.plot(figsize=(10, 10), alpha=0.5, edgecolor='k')
        cx.add_basemap(ax, zoom=12, crs=sensors_gdf.crs)
        ax.set_title("Sensor Locations for PEMS-BAY")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        if idx:
            pass
        plt.show()

    @staticmethod
    def _get_adj_mx(sensor_ids, distance_df, threshold):
        """Creates an adjacency matrix from a DataFrame of distances.
        
        Adapted from: https://github.com/dmlc/dgl/blob/master/examples/pytorch/stgcn_wave/sensors2graph.py
        """
        sensor_ids.sort()
        # Create mxm distance matrix
        n_sensors = sensor_ids.shape[0]
        dists = np.full((n_sensors, n_sensors), np.inf)
        # Map sensor IDs to indices
        sid_to_idx = {sid: i for i, sid in enumerate(sensor_ids)}
        # Collect distances between sensors
        for src, dst, dist in distance_df.values:
            if src in sid_to_idx and dst in sid_to_idx:
                dists[sid_to_idx[src], sid_to_idx[dst]] = dist
        # Create an adjacency matrix by inverting the normalized distances between sensors.
        stddev = dists[~np.isinf(dists)].std()
        adj = np.exp(-(dists / stddev)**2)
        # Zero out any small values to make `adj` sparse
        adj[adj < threshold] = 0
        # Sparsify and Symmetrize
        adj = coo_matrix(adj)
        adj = (adj + adj.T) / 2
        return adj

    def _get_edges(self, adj):
        edge_idx = []
        edge_wt = []
        for i in range(self.num_vertices):
            for j in range(self.num_vertices):
                if adj[i, j] > 0:
                    edge_idx.append((i, j))
                    edge_wt.append(adj[i, j])
        edge_idx = torch.tensor(edge_idx, dtype=torch.long).t().contiguous()
        edge_wt = torch.tensor(edge_wt,  dtype=torch.float)
        return edge_idx, edge_wt

    def _generate_splits(self, traffic_file):
        train_pct, val_pct, _ = self.split_sizes
        traffic_df = pd.read_hdf(traffic_file)
        traffic_df = traffic_df.reset_index().drop(columns=['index'])
        train_end = int(len(traffic_df) * train_pct)
        val_end = int(train_end + len(traffic_df) * val_pct)
        train, val, test = traffic_df[:train_end], traffic_df[train_end:val_end], traffic_df[val_end:]
        for split in ('train', 'val', 'test'):
            eval(split).to_hdf(os.path.join(self.data_dir, f"{split}.h5"), key=f"{split}_data", index=False)

    def _create_windows(self, X_raw: np.ndarray, hist_window: int, pred_window: int):
        m, _ = X_raw.shape
        n_windows = m // (hist_window + pred_window)
        self.X = torch.zeros((n_windows, self.num_vertices, hist_window))
        self.y = torch.zeros((n_windows, self.num_vertices, pred_window))
        for i in range(n_windows):
            start = i * (hist_window + pred_window)
            train_end = start + hist_window
            pred_end = train_end + pred_window
            self.X[i, :] = torch.tensor(X_raw[start:train_end].T)
            self.y[i, :] = torch.tensor(X_raw[train_end:pred_end].T)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx, :].reshape(1, self.num_vertices, -1), self.y[idx, :].reshape(1, self.num_vertices, -1)
