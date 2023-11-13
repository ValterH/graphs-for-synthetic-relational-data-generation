import pathlib
import numpy as np
import pandas as pd
import torch

###########################################################################################
# code reference: https://pytorch-geometric.readthedocs.io/en/latest/tutorial/load_csv.html
# dataset reference: https://www.kaggle.com/competitions/rossmann-store-sales/data
###########################################################################################

FILE_ABS_PATH = pathlib.Path(__file__) # absolute path of this file

###########################################################################################
###########################################################################################

# TODO: date&time encoding
# TODO: handle the target variable

# TODO: problems with missing values when encoding
# maybe add a parameter which type of data we're dealing with and what to do with missing values to handle this when encoding
# TODO: these operations could be vectorized - faster than using for loops
# TODO: better error handling
# TODO: fix so we dont map NaN's when encoding categories


class CategoricalEncoder:
    def __init__(self):
        pass
    
    def __call__(self, series):
        mapping = {category: i for i, category in enumerate(series.unique())}

        x = torch.zeros(len(series), 1)
        for i, category in enumerate(series.values):
            try:
                x[i, 0] = mapping[category]
            except Exception:
                x[i, 0] = -1
        return x


class OneHotEncoder:
    def __init__(self):
        pass
    
    def __call__(self, series):
        categories = np.array(series.unique())
        
        x = torch.zeros(len(series), len(categories))
        for i, category in enumerate(series.values):
            try:
                x[i, :] = torch.from_numpy(np.isin(categories, category).astype(int))
            except Exception:
                x[i, :] = torch.from_numpy(np.zeros(len(categories)).astype(int))
        return x


# TODO: maybe specify if we want floats or integers
class NumericalEncoder:
    def __init__(self):
        pass
    
    def __call__(self, series):
        return torch.tensor(series.values, dtype=torch.float).view(-1, 1)


# NOTE: this acts like a one-hot encoder, but can have multiple 1s in a row
# TODO: maybe write a more general class for this types of encoders?
class PromoIntervalEncoder:
    def __init__(self, sep=","):
        self.sep = sep
        self.months = np.array(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sept", "Oct", "Nov", "Dec"])
    
    def __call__(self, series):
        x = torch.zeros(len(series), 12)
        for i, months in enumerate(series.values):
            try:
                x[i, :] = torch.from_numpy(np.isin(self.months, np.array(months.split(self.sep))).astype(int))
            except Exception:
                x[i, :] = torch.from_numpy(np.zeros(12).astype(int))
        return x


class IdentityEncoder:
    def __init__(self, dtype=None):
        self.dtype = dtype

    def __call__(self, df):
        return torch.from_numpy(df.values).view(-1, 1).to(self.dtype)


###########################################################################################
###########################################################################################


def load_node_csv(path, index_col, encoders=None, **kwargs):
    df = pd.read_csv(path, index_col=index_col, **kwargs)
    mapping = {index: i for i, index in enumerate(df.index.unique())}

    x = None
    if encoders is not None:
        xs = [encoder(df[col]) for col, encoder in encoders.items()]
        x = torch.cat(xs, dim=-1)

    return x, mapping


def load_edge_csv(path, src_index_col, src_mapping, dst_index_col, dst_mapping, encoders=None, **kwargs):
    df = pd.read_csv(path, **kwargs)
    
    # NOTE: since the "sales" df (train.csv) has no index column copy the default index to a column named "index"
    df["index"] = df.index

    src = [src_mapping[index] for index in df[src_index_col]]
    dst = [dst_mapping[index] for index in df[dst_index_col]]
    edge_index = torch.tensor([src, dst])

    edge_attr = None
    if encoders is not None:
        edge_attrs = [encoder(df[col]) for col, encoder in encoders.items()]
        edge_attr = torch.cat(edge_attrs, dim=-1)

    return edge_index, edge_attr


###########################################################################################
###########################################################################################


def main():
    pass

if __name__ == "__main__":
    main()
