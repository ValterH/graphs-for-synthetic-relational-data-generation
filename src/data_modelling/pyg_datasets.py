import torch
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

_ROSSMANN_FEATURES_TO_KEEP_GIN = ["type", "degree"]
_ROSSMANN_FEATURE_MAPPING_GIN = {"type": {"store": 0, "sale": 1}}
_MUTAGENESIS_FEATURES_TO_KEEP_GIN = ["type", "degree"]
_MUTAGENESIS_FEATURE_MAPPING_GIN = {"type": {"molecule": 0, "atom": 1, "bond": 2}}

def get_rossmann_dataset(root="data/rossmann/pyg", features=_ROSSMANN_FEATURES_TO_KEEP_GIN, feature_mappings=_ROSSMANN_FEATURE_MAPPING_GIN, target=True):
    
    G = get_rossmann_graph(root_nodes=False, features=features, feature_mappings=feature_mappings)
    rossmann_data_list = [from_networkx(G, group_node_attrs=features)]
    # we assume that the target is the last feature
    # from_networkx doesn't support specifying the target so we have to do it manually
    if target:
        for data in rossmann_data_list:
            data.y = data.x[:, -1].view(-1, 1).type(torch.float)
            data.x = data.x[:, :-1].type(torch.float)
    
    # TODO: if we already have saved files this will not update them
    # process() should do it afaik but its not working ...
    rossmann = Rossmann(root=root, data_list=rossmann_data_list)
    rossmann.process()
    return rossmann


def get_rossmann_subgraphs_dataset(root="data/rossmann/pyg_subgraphs", features=_ROSSMANN_FEATURES_TO_KEEP_GIN, feature_mappings=_ROSSMANN_FEATURE_MAPPING_GIN, target=True):
    
    subgraphs = get_rossmann_subgraphs(features=features, feature_mappings=feature_mappings)
    rossmann_data_list = [from_networkx(G, group_node_attrs=features) for G in  subgraphs]
    
    # we assume that the target is the last feature
    # from_networkx doesn't support specifying the target so we have to do it manually
    if target:
        for data in rossmann_data_list:
            data.y = data.x[:, -1].view(-1, 1).type(torch.float)
            data.x = data.x[:, :-1].type(torch.float)
    
    # TODO: if we already have saved files this will not update them
    # process() should do it afaik but its not working ...
    rossmann = Rossmann(root=root, data_list=rossmann_data_list)
    rossmann.process()
    return rossmann


def get_mutagenesis_dataset(root="data/mutagenesis/pyg", features=_MUTAGENESIS_FEATURES_TO_KEEP_GIN, feature_mappings=_MUTAGENESIS_FEATURE_MAPPING_GIN, target=True):
    
    G = get_mutagenesis_graph(root_nodes=False, features=features, feature_mappings=feature_mappings)
    mutagenesis_data_list = [from_networkx(G, group_node_attrs=features)]
    
    # we assume that the target is the last feature
    # from_networkx doesn't support specifying the target so we have to do it manually
    if target:
        for data in mutagenesis_data_list:
            data.y = data.x[:, -1].view(-1, 1).type(torch.float)
            data.x = data.x[:, :-1].type(torch.float)
    
    # TODO: if we already have saved files this will not update them
    # process() should do it afaik but its not working ...
    mutagenesis = Mutagenesis(root=root, data_list=mutagenesis_data_list)
    mutagenesis.process()
    return mutagenesis


def get_mutagenesis_subgraphs_dataset(root="data/mutagenesis/pyg_subgraphs", features=_MUTAGENESIS_FEATURES_TO_KEEP_GIN, feature_mappings=_MUTAGENESIS_FEATURE_MAPPING_GIN, target=True):
    
    subgraphs = get_mutagenesis_subgraphs(features=features, feature_mappings=feature_mappings)
    mutagenesis_data_list = [from_networkx(G, group_node_attrs=features) for G in subgraphs]
    
    # we assume that the target is the last feature
    # from_networkx doesn't support specifying the target so we have to do it manually
    if target:
        for data in mutagenesis_data_list:
            data.y = data.x[:, -1].view(-1, 1).type(torch.float)
            data.x = data.x[:, :-1].type(torch.float)
    
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
