import contextily as cx
import geopandas
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
from scipy.sparse import coo_matrix


def draw_road_graph(adj: coo_matrix, pos: dict, ax=None, **kwargs):
    # Create figure
    if ax is None:
        plt.figure(figsize=kwargs.get('figsize', (8, 6)), facecolor='white')
        ax = plt.gca()
    ax.set_title(kwargs.get('title', "Road Graph"), fontsize=18)
    # Build a `networkx.Graph` using `adj`.
    v, _ = adj.shape
    ns = tuple(pos.keys())
    G = nx.Graph()
    for i in range(v):
        for j in range(v):
            G.add_edge(ns[i], ns[j], weight=adj[i, j])
    # Define the width of each edge based on its weight
    width = list(nx.get_edge_attributes(G, 'weight').values())
    # Draw the `networkx.Graph`
    nx.draw_networkx(
        G,
        pos=pos,
        with_labels=False,
        node_size=kwargs.get('node_size', 40),
        width=width,
        edgecolors=kwargs.get('edgecolors', 'k'),
        linewidths=kwargs.get('linewidths', 0.5),
        ax=ax
    )
    # Add longitude and latitude labels if provided
    if 'xticks' in kwargs or 'yticks' in kwargs:
        ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
        ax.set_xticks(kwargs['xticks'])
        ax.set_yticks(kwargs['yticks'])
        ax.set_xlabel("Longitude", fontsize=14)
        ax.set_ylabel("Latitude", fontsize=14)


def draw_traffic_plot(gdf: geopandas.GeoDataFrame, traffic, ax=None, **kwargs):
    if ax is None:
        plt.figure(figsize=kwargs.get('figsize', (8, 6)), facecolor='white')
        ax = plt.gca()
    # Draw traffic data
    gdf['traffic'] = traffic.flatten()
    gdf.plot(
        ax=ax,
        alpha=kwargs.get('alpha', 0.7),
        edgecolor=kwargs.get('edgecolor', 'k'),
        column='traffic',
        cmap=kwargs.get('cmap', 'RdYlGn'),
        vmin=kwargs.get('vmin'),
        vmax=kwargs.get('vmax')
    )
    cx.add_basemap(ax, zoom=12, crs=gdf.crs, source=cx.providers.OpenStreetMap.Mapnik)
    # Label axes
    ax.set_title(kwargs.get('title') or "Traffic", fontsize=kwargs.get('titlesize', 18))
    ax.set_xlabel("Longitude", fontsize=kwargs.get('labelsize', 14))
    ax.set_ylabel("Latitude", fontsize=kwargs.get('labelsize', 14))
    # Add colorbar
    fig = plt.gcf()
    norm = matplotlib.colors.Normalize(vmin=kwargs.get('vmin', traffic.min()), vmax=kwargs.get('vmax', traffic.max()))
    mpbl = matplotlib.cm.ScalarMappable(norm, cmap=kwargs.get('cmap', 'RdYlGn'))
    fig.colorbar(
        mpbl,
        ax=ax,
        orientation='vertical',
        shrink=kwargs.get('shrink', 0.7),
        aspect=kwargs.get('aspect', 25),
        label=kwargs.get('cbar_label', 'Speed (mph)')
    )
