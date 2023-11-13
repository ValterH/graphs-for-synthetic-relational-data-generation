import pathlib
import numpy as np
import pandas as pd
import networkx as nx
from torch_geometric.utils.convert import to_networkx, from_networkx
from torch_geometric.data import Dataset, HeteroData

###########################################################################################

FILE_ABS_PATH = pathlib.Path(__file__) # absolute path of this file

###########################################################################################

# TODO: I don't think we want weakly connected components here... think about it more
# def rossmann_to_nx_components(dir_path, train=True):
#     G, _ = rossmann_to_nx_graph(dir_path, train)
#     Gs = []
#     for weekly_connected_component in nx.weakly_connected_components(G):
#         # create a subgraph for each connected component
#         Gs.append(G.subgraph(weekly_connected_component))
#     return Gs

def rossmann_to_nx_components(dir_path, parent_nodes):
    G = rossmann_to_nx_graph(dir_path)
    Gs = []
    for parent_node in parent_nodes:
        # TODO: 1 graph for each node in the parent table since we're doing 1:N
        pass
    return Gs


def rossmann_to_nx_graph(dir_path, train=True):
    # paths of the csv files
    dir_path = pathlib.Path(dir_path)
    store_path = dir_path / "store.csv"
    if train:
        sales_path = dir_path / "train.csv"
    else:
        sales_path = dir_path / "test.csv"
    
    store_df = pd.read_csv(store_path, index_col="Store")
    
    # assign a unique index to each sale
    # we need it to be different from the store index
    sales_df = pd.read_csv(sales_path, index_col=None)
    sales_df["Sale"] = sales_df.index + store_df.index.max() + 1
    sales_mapping = {sale: i for i, sale in enumerate(sales_df["Sale"])}
    
    # create a graph from the indices in the source and target columns
    G = nx.from_pandas_edgelist(sales_df, source="Store", target="Sale", edge_attr=None, create_using=nx.DiGraph)
    
    # TODO: add node attributes
    
    return G, sales_mapping

###########################################################################################

# TODO: weap in a Dataset class
class RossmanDataset(Dataset):
    pass

###########################################################################################

def main():
    dir_path = FILE_ABS_PATH.parent.parent.parent / "data" / "rossmann"
    Gs = rossmann_to_nx_components(dir_path)


if __name__ == "__main__":
    main()
