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

_ROSSMANN_FEATURES_TO_KEEP_GIN = ["y"]
_ROSSMANN_FEATURE_MAPPING_GIN = {"y": {"store": 0, "sale": 1}}
_MUTAGENESIS_FEATURES_TO_KEEP_GIN = ["y"]
_MUTAGENESIS_FEATURE_MAPPING_GIN = {"y": {"molecule": 0, "atom": 1, "bond": 2}}

def get_rossmann_dataset(root="data/rossmann/pyg", features=_ROSSMANN_FEATURES_TO_KEEP_GIN, feature_mappings=_ROSSMANN_FEATURE_MAPPING_GIN):
    
    G = get_rossmann_graph(root_nodes=False, features=features, feature_mappings=feature_mappings)
    rossmann_data_list = [from_networkx(G, group_node_attrs=features)]
    
    return Rossmann(root=root, data_list=rossmann_data_list)


def get_rossmann_subgraphs_dataset(root="data/rossmann/pyg_subgraphs", features=_ROSSMANN_FEATURES_TO_KEEP_GIN, feature_mappings=_ROSSMANN_FEATURE_MAPPING_GIN):
    
    subgraphs = get_rossmann_subgraphs(features=features, feature_mappings=feature_mappings)
    rossmann_data_list = [from_networkx(G, group_node_attrs=features) for G in  subgraphs]
    
    return Rossmann(root=root, data_list=rossmann_data_list)


def get_mutagenesis_dataset(root="data/mutagenesis/pyg", features=_MUTAGENESIS_FEATURES_TO_KEEP_GIN, feature_mappings=_MUTAGENESIS_FEATURE_MAPPING_GIN):
        
        G = get_mutagenesis_graph(root_nodes=False, features=features, feature_mappings=feature_mappings)
        mutagenesis_data_list = [from_networkx(G, group_node_attrs=features)]
        
        return Mutagenesis(root=root, data_list=mutagenesis_data_list)


def get_mutagenesis_subgraphs_dataset(root="data/mutagenesis/pyg_subgraphs", features=_MUTAGENESIS_FEATURES_TO_KEEP_GIN, feature_mappings=_MUTAGENESIS_FEATURE_MAPPING_GIN):
    
    subgraphs = get_mutagenesis_subgraphs(features=features, feature_mappings=feature_mappings)
    mutagenesis_data_list = [from_networkx(G, group_node_attrs=features) for G in subgraphs]
    
    return Mutagenesis(root=root, data_list=mutagenesis_data_list)

############################################################################################

def main():
    pass

if __name__ == "__main__":
    main()
