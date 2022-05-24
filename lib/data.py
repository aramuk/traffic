import os
from typing import Tuple

import contextily as cx
import geopandas
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class PEMSBay(Dataset):
    def __init__(self, data_dir: str, splits: Tuple[float], window_size):
        self.data_dir = data_dir
        dist_file = os.path.join(data_dir, 'distances.csv')
        sensors_file = os.path.join(data_dir, 'sensor_locations.csv')
        traffic_file = os.path.join(data_dir, 'traffic.h5')
        if not all(os.path.exists(f) for f in (dist_file, sensors_file, traffic_file)):
            raise FileNotFoundError(f"Missing dataset files: {dist_file}, {sensors_file}, or {traffic_file}")

        self.dist_df = pd.read_csv(dist_file, names=['from', 'to', 'distance'])
        self.sensors_df = pd.read_csv(sensors_file, names=['sensor_id', 'latitude', 'longitude'])
        self.adj = self._get_adj_mx()
        traffic_df = pd.read_hdf(traffic_file)
        assert len(splits) == 3, f"Expected 3 splits, received {len(splits)}"
        self.train_size, self.val_size, self.test_size = splits
        

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

    def _get_adj_mx(self, K=0.1):
        """Creates an adjacency matrix from a DataFrame of distances.
        
        Adapted from: https://github.com/dmlc/dgl/blob/master/examples/pytorch/stgcn_wave/sensors2graph.py
        """
        sensor_ids = self.sensors_df.sensor_id.unique()
        sensor_ids.sort()
        # Create mxm distance matrix
        n_sensors = sensor_ids.shape[0]
        dists = np.full((n_sensors, n_sensors), np.inf)
        # Map sensor IDs to indices
        sid_to_idx = {sid: i for i, sid in enumerate(sensor_ids)}
        # Collect distances between sensors
        for src, dst, dist in self.distance_df.values:
            if src in sid_to_idx and dst in sid_to_idx:
                dists[sid_to_idx[src], sid_to_idx[dst]] = dist
        # Create an adjacency matrix by inverting the normalized distances between sensors.
        stddev = dists[~np.isinf(dists)].std()
        adj = np.exp(-(dists / stddev)**2)
        # Zero out any small values to make `adj` sparse
        adj[adj < K] = 0
        return adj

    def _transform(self):
        pass

    def __len__(self):
        pass

    def __getitem__(self):
        pass