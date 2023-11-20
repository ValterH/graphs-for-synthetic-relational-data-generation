import torch
from torch_geometric.utils import from_networkx
from torch_geometric.data import InMemoryDataset
from src.data_modelling.table_to_graph import graph_to_subgraphs

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


def get_rossmann_dataset():
    from src.data_modelling.table_to_graph import ROSSMANN_GRAPH, ROSSMANN_ROOT_NODES
    rossmann_data_list = [from_networkx(subgraph) for subgraph in graph_to_subgraphs(ROSSMANN_GRAPH, ROSSMANN_ROOT_NODES)]
    return Rossmann(root="data/rossmann", data_list=rossmann_data_list)

def get_mutagenesis_dataset():
    from src.data_modelling.table_to_graph import MUTAGENESIS_GRAPH, MUTAGENESIS_ROOT_NODES
    mutagenesis_data_list = [from_networkx(subgraph) for subgraph in graph_to_subgraphs(MUTAGENESIS_GRAPH, MUTAGENESIS_ROOT_NODES)]
    return Mutagenesis(root="data/mutagenesis", data_list=mutagenesis_data_list)

############################################################################################

def main():
    rossmann = get_rossmann_dataset()
    pass

if __name__ == "__main__":
    main()
