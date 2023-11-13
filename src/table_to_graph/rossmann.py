import pathlib
import numpy as np
import pandas as pd
from torch_geometric.data import HeteroData

# absolute import (installed package)
from src.table_to_graph.utils import (
    CategoricalEncoder, OneHotEncoder, NumericalEncoder, PromoIntervalEncoder,
    load_node_csv, load_edge_csv
)
# TODO: can't get relative import to work
# from .utils import (
#     CategoricalEncoder, OneHotEncoder, NumericalEncoder, PromoIntervalEncoder,
#     load_node_csv, load_edge_csv
# )

###########################################################################################
# code reference: https://pytorch-geometric.readthedocs.io/en/latest/tutorial/load_csv.html
# dataset reference: https://www.kaggle.com/competitions/rossmann-store-sales/data
###########################################################################################

FILE_ABS_PATH = pathlib.Path(__file__) # absolute path of this file

###########################################################################################
###########################################################################################


def rossmann_to_graph(dir_path):
    # paths of the csv files
    dir_path = pathlib.Path(dir_path)
    store_path = dir_path / "store.csv"
    sales_train_path = dir_path / "train.csv"
    test_path = dir_path / "test.csv"
    
    # create nodes for stores
    store_x, store_mapping = load_node_csv(
        store_path, index_col="Store", encoders={
        "StoreType": CategoricalEncoder(),
        "Assortment": CategoricalEncoder(),
        "CompetitionDistance": NumericalEncoder(),
        "CompetitionOpenSinceMonth": CategoricalEncoder(),
        "CompetitionOpenSinceYear": CategoricalEncoder(), # TODO: think about this - it should be integers from some year onwards?
        "Promo2": CategoricalEncoder(),
        "Promo2SinceWeek": CategoricalEncoder(), # TODO: think about this
        "Promo2SinceYear": CategoricalEncoder(), # TODO: think about this - it should be integers from some year onwards?
        "PromoInterval": PromoIntervalEncoder(),
    })
    
    # create nodes for sales
    sales_x, sales_mapping = load_node_csv(
        sales_train_path, index_col=None, encoders={
        "DayOfWeek": CategoricalEncoder(), 
        # "Date": CategoricalEncoder(), # TODO: how should we encode dates
        "Sales": NumericalEncoder(),
        "Customers": NumericalEncoder(),
        "Open": CategoricalEncoder(),
        "Promo": CategoricalEncoder(),
        "StateHoliday": CategoricalEncoder(),
        "SchoolHoliday": CategoricalEncoder(),
    })
    
    # create edge index
    edge_index, edge_label = load_edge_csv(
        sales_train_path,
        src_index_col='Store',
        src_mapping=store_mapping,
        dst_index_col="index",
        dst_mapping=sales_mapping,
        encoders=None # if we would want edge features
    )
    
    # create empty graph
    data = HeteroData()
    # put the nodes into a graph
    data["store"].x = store_x
    data["sales"].x = sales_x
    # connect the nodes with edges
    data["store", "sells", "sales"].edge_index = edge_index
    data["store", "sells", "sales"].edge_label = edge_label
    
    return data


###########################################################################################
###########################################################################################


def main():
    # root/src/table_to_graph/this_file.py
    # root/data/rossmann
    dir_path = FILE_ABS_PATH.parent.parent.parent / "data" / "rossmann"
    data = rossmann_to_graph(dir_path)

if __name__ == "__main__":
    main()
