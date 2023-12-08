import random

import torch
import networkx as nx
from torch_geometric.utils import from_networkx
from torch_geometric.data import InMemoryDataset

from src.data_modelling.table_to_graph import (
    get_rossmann_graph,
    get_rossmann_subgraphs, 
    get_mutagenesis_graph,
    get_mutagenesis_subgraphs
)

############################################################################################
# reference: https://pytorch-geometric.readthedocs.io/en/latest/tutorial/create_dataset.html
############################################################################################

class MyDataset(InMemoryDataset):
    def __init__(self, root, data_list, transform=None):
        self.data_list = data_list
        super().__init__(root, transform)
        self.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return 'data.pt'

    def process(self):
        self.save(self.data_list, self.processed_paths[0])


class Rossmann(MyDataset):
    def __init__(self, root, data_list, transform=None):
        super().__init__(root, data_list, transform)


class Mutagenesis(MyDataset):
    def __init__(self, root, data_list, transform=None):
        super().__init__(root, data_list, transform)


############################################################################################

_ROSSMANN_FEATURES_TO_KEEP_GIN = ["index", "type", "k-hop_degrees"]
_ROSSMANN_FEATURES_LENGTH = 1
_ROSSMANN_FEATURE_MAPPING_GIN = {"type": {"store": 0, "sale": 1}}

_MUTAGENESIS_FEATURES_TO_KEEP_GIN = ["index", "type", "k-hop_degrees"]
_MUTAGENESIS_FEATURES_LENGTH = 1
_MUTAGENESIS_FEATURE_MAPPING_GIN = {"type": {"molecule": 0, "atom": 1, "bond": 2}}


def sample_relational_distribution(dataset_name, num_graphs, seed=42):
    if dataset_name == "rossmann-store-sales":
        features = _ROSSMANN_FEATURES_TO_KEEP_GIN
        features_length = _ROSSMANN_FEATURES_LENGTH
        dataset_class = Rossmann
        root = "data/pyg/rossmann"
        subgraphs = get_rossmann_subgraphs(features=features, feature_mappings=_ROSSMANN_FEATURE_MAPPING_GIN)
    elif dataset_name == "mutagenesis":
        features = _MUTAGENESIS_FEATURES_TO_KEEP_GIN
        features_length = _MUTAGENESIS_FEATURES_LENGTH
        dataset_class = Mutagenesis
        root = "data/pyg/mutagenesis"
        subgraphs = get_mutagenesis_subgraphs(features=features, feature_mappings=_MUTAGENESIS_FEATURE_MAPPING_GIN)
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")
    random.seed(seed)
    sampled_subgraphs = random.sample(subgraphs, num_graphs)
    G = sampled_subgraphs[0]
    for i in range(1, len(sampled_subgraphs)):
        G = nx.disjoint_union(G, sampled_subgraphs[i])
    data_list = [from_networkx(G, group_node_attrs=features)]
    return from_data_list(data_list, dataset_class=dataset_class, root=root, features_length=features_length, target=False)


def from_data_list(data_list, dataset_class=Rossmann, root="data/pyg/rossmann", features_length=_ROSSMANN_FEATURES_LENGTH, target=True):
    # we assume that the target is the last feature
    # from_networkx doesn't support specifying the target so we have to do it manually
    if target:
        for data in data_list:
            # we assume that the first column is the index (needed for future mappings)
            data.index = data.x[:, 0]
            # we assume that the target is located at the last few columns
            data.y = data.x[:, (features_length + 1):].view(-1, data.x.shape[1] - features_length - 1).type(torch.float)
            # we assume the features are located in the middle columns
            data.x = data.x[:, 1:(features_length + 1)].type(torch.float)
    
    # TODO: if we already have saved files this will not update them
    # process() should do it afaik but its not working ...
    dataset = dataset_class(root=root, data_list=data_list)
    dataset.process()
    return dataset


def get_rossmann_dataset(root="data/pyg/rossmann", features=_ROSSMANN_FEATURES_TO_KEEP_GIN, features_length=_ROSSMANN_FEATURES_LENGTH, feature_mappings=_ROSSMANN_FEATURE_MAPPING_GIN, target=True):
    
    G = get_rossmann_graph(root_nodes=False, features=features, feature_mappings=feature_mappings)
    rossmann_data_list = [from_networkx(G, group_node_attrs=features)]
    # we assume that the target is the last feature
    # from_networkx doesn't support specifying the target so we have to do it manually
    if target:
        for data in rossmann_data_list:
            # we assume that the first column is the index (needed for future mappings)
            data.index = data.x[:, 0]
            # we assume that the target is located at the last few columns
            data.y = data.x[:, (features_length + 1):].view(-1, data.x.shape[1] - features_length - 1).type(torch.float)
            # we assume the features are located in the middle columns
            data.x = data.x[:, 1:(features_length + 1)].type(torch.float)
    
    # TODO: if we already have saved files this will not update them
    # process() should do it afaik but its not working ...
    rossmann = Rossmann(root=root, data_list=rossmann_data_list)
    rossmann.process()
    return rossmann


def get_rossmann_subgraphs_dataset(root="data/pyg/rossmann_subgraphs", features=_ROSSMANN_FEATURES_TO_KEEP_GIN, features_length=_ROSSMANN_FEATURES_LENGTH, feature_mappings=_ROSSMANN_FEATURE_MAPPING_GIN, target=True):
    
    subgraphs = get_rossmann_subgraphs(features=features, feature_mappings=feature_mappings)
    rossmann_data_list = [from_networkx(G, group_node_attrs=features) for G in  subgraphs]
    
    # we assume that the target is the last feature
    # from_networkx doesn't support specifying the target so we have to do it manually
    if target:
        for data in rossmann_data_list:
            # we assume that the first column is the index (needed for future mappings)
            data.index = data.x[:, 0]
            # we assume that the target is located at the last few columns
            data.y = data.x[:, (features_length + 1):].view(-1, data.x.shape[1] - features_length - 1).type(torch.float)
            # we assume the features are located in the middle columns
            data.x = data.x[:, 1:(features_length + 1)].type(torch.float)
    
    # TODO: if we already have saved files this will not update them
    # process() should do it afaik but its not working ...
    rossmann = Rossmann(root=root, data_list=rossmann_data_list)
    rossmann.process()
    return rossmann


def get_mutagenesis_dataset(root="data/pyg/mutagenesis", features=_MUTAGENESIS_FEATURES_TO_KEEP_GIN, features_length=_MUTAGENESIS_FEATURES_LENGTH, feature_mappings=_MUTAGENESIS_FEATURE_MAPPING_GIN, target=True):
    
    G = get_mutagenesis_graph(root_nodes=False, features=features, feature_mappings=feature_mappings)
    mutagenesis_data_list = [from_networkx(G, group_node_attrs=features)]
    
    # we assume that the target is the last feature
    # from_networkx doesn't support specifying the target so we have to do it manually
    if target:
        for data in mutagenesis_data_list:
            # we assume that the first column is the index (needed for future mappings)
            data.index = data.x[:, 0]
            # we assume that the target is located at the last few columns
            data.y = data.x[:, (features_length + 1):].view(-1, data.x.shape[1] - features_length - 1).type(torch.float)
            # we assume the features are located in the middle columns
            data.x = data.x[:, 1:(features_length + 1)].type(torch.float)
    
    # TODO: if we already have saved files this will not update them
    # process() should do it afaik but its not working ...
    mutagenesis = Mutagenesis(root=root, data_list=mutagenesis_data_list)
    mutagenesis.process()
    return mutagenesis


def get_mutagenesis_subgraphs_dataset(root="data/pyg_subgraphs/mutagenesis", features=_MUTAGENESIS_FEATURES_TO_KEEP_GIN, features_length=_MUTAGENESIS_FEATURES_LENGTH, feature_mappings=_MUTAGENESIS_FEATURE_MAPPING_GIN, target=True):
    
    subgraphs = get_mutagenesis_subgraphs(features=features, feature_mappings=feature_mappings)
    mutagenesis_data_list = [from_networkx(G, group_node_attrs=features) for G in subgraphs]
    
    # we assume that the target is the last feature
    # from_networkx doesn't support specifying the target so we have to do it manually
    if target:
        for data in mutagenesis_data_list:
            # we assume that the first column is the index (needed for future mappings)
            data.index = data.x[:, 0]
            # we assume that the target is located at the last few columns
            data.y = data.x[:, (features_length + 1):].view(-1, data.x.shape[1] - features_length - 1).type(torch.float)
            # we assume the features are located in the middle columns
            data.x = data.x[:, 1:(features_length + 1)].type(torch.float)
    
    # TODO: if we already have saved files this will not update them
    # process() should do it afaik but its not working ...
    mutagenesis = Mutagenesis(root=root, data_list=mutagenesis_data_list)
    mutagenesis.process()
    return mutagenesis

############################################################################################

def main():
    pass

if __name__ == "__main__":
    main()
