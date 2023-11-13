import pathlib
import numpy as np
import pandas as pd
import networkx as nx
from torch_geometric.utils.convert import to_networkx, from_networkx
from torch_geometric.data import Dataset, HeteroData

###########################################################################################

FILE_ABS_PATH = pathlib.Path(__file__) # absolute path of this file

###########################################################################################

# with the assumptions that we're dealing with a 1:N relationship 
# we can specify the parent nodes and get the connected components by looking at the descendants
def rossmann_to_nx_components(G, parent_nodes):
    Gs = []
    for parent_node in parent_nodes:
        reachable_nodes = nx.descendants(G, parent_node).add(parent_node)
        Gs.append(G.subgraph(reachable_nodes))
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
    
    # get each component of the graph
    Gs = rossmann_to_nx_components(G, store_df.index)
    
    return G, Gs



###########################################################################################

# TODO: weap in a Dataset class
class RossmanDataset(Dataset):
    pass

###########################################################################################

def main():
    dir_path = FILE_ABS_PATH.parent.parent.parent / "data" / "rossmann"
    G, Gs = rossmann_to_nx_graph(dir_path)
    pass


if __name__ == "__main__":
    main()
