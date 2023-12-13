import os
import shutil
import random

import torch
import networkx as nx
from torch_geometric.utils import from_networkx
from torch_geometric.data import InMemoryDataset

from src.data_modelling.table_to_graph import database_to_graph, database_to_subgraphs
from src.data_modelling.feature_engineering import add_index, add_k_hop_degrees, filter_graph_features_with_mapping

############################################################################################
# reference: https://pytorch-geometric.readthedocs.io/en/latest/tutorial/create_dataset.html
############################################################################################

class GraphRelationalDataset(InMemoryDataset):
    def __init__(self, root, data_list, transform=None):
        self.data_list = data_list
        super().__init__(root, transform)
        self.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return 'data.pt'

    def process(self):
        if os.path.exists(self.root):
            shutil.rmtree(self.root)
        self.save(self.data_list, self.processed_paths[0])


############################################################################################


def sample_relational_distribution(dataset_name, num_graphs, seed=42):
    if dataset_name == "rossmann-store-sales":
        features = _ROSSMANN_FEATURES_TO_KEEP_GIN
        features_length = _ROSSMANN_FEATURES_LENGTH
        dataset_class = Rossmann
        root = "data/pyg/rossmann_sample"
        subgraphs = get_rossmann_subgraphs(features=features, feature_mappings=_ROSSMANN_FEATURE_MAPPING_GIN)
    elif dataset_name == "mutagenesis":
        features = _MUTAGENESIS_FEATURES_TO_KEEP_GIN
        features_length = _MUTAGENESIS_FEATURES_LENGTH
        dataset_class = Mutagenesis
        root = "data/pyg/mutagenesis_sample"
        subgraphs = get_mutagenesis_subgraphs(features=features, feature_mappings=_MUTAGENESIS_FEATURE_MAPPING_GIN)
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")
    
    # set seed and sample with replacement
    random.seed(seed)
    sampled_subgraphs = random.choices(subgraphs, k = num_graphs)
    
    # avoid relabeling by using a list of graphs
    sampled_subgraphs = [from_networkx(G, group_node_attrs=features) for G in sampled_subgraphs]
    previous_graph_nodes = 0
    for i in range(len(sampled_subgraphs)):
        sampled_subgraphs[i].edge_index += previous_graph_nodes
        previous_graph_nodes += len(sampled_subgraphs[i].x)
    # concat all of the tensors
    all_graphs_data = sampled_subgraphs[0]
    for data in sampled_subgraphs[1:]:
        all_graphs_data.edge_index = torch.cat((all_graphs_data.edge_index, data.edge_index), dim=1)
        all_graphs_data.x = torch.cat((all_graphs_data.x, data.x), dim=0)
    all_graphs_data.index = torch.tensor(range(all_graphs_data.x.shape[0]))
    all_graphs_data.x = all_graphs_data.x[:, 1:(features_length + 1)].type(torch.float)
    data_list = [all_graphs_data]    
    
    if os.path.exists(root):
        shutil.rmtree(root)
    
    dataset = dataset_class(root=root, data_list=data_list)
    dataset.process()
    return dataset


def create_pyg_dataset(dataset_name, pyg_dataset_save_path, features, feature_mappings, target):
    
    # load the graph representing the database
    G, _ = database_to_graph(dataset_name)
    
    G = add_index(G)
    if target == "k_hop_degrees":
        G = add_k_hop_degrees(G, k=2)
        target_length = 2
    else:
        raise ValueError(f"Target {target} not supported")
    
    features_to_keep = ["index", *features, target]
    G = filter_graph_features_with_mapping(G, features_to_keep, feature_mappings)
    
    # convert to pytorch geometric Data object
    data = from_networkx(G, group_node_attrs=features_to_keep)
    
    # because they don't support specifying node targets or arbitrary arguments we have to relabel manually
    nrows, ncols = data.x.shape
    data.index = data.x[:, 0]
    data.y = data.x[:, (ncols - target_length):].type(torch.float)
    data.x = data.x[:, 1:(ncols - target_length)].type(torch.float)
    
    
    dataset = GraphRelationalDataset(root=pyg_dataset_save_path, data_list=[data])
    dataset.process()
    return dataset


############################################################################################

def main():
    rossmann_feature_mappings = {"type": {"store": 0, "sale": 1}}
    rossmann_dataset = create_pyg_dataset("rossmann-store-sales", "data/pyg/rossmann", ["type"], rossmann_feature_mappings, "k_hop_degrees")
    
    mutagenesis_feature_mappings = {"type": {"molecule": 0, "atom": 1, "bond": 2}}
    mutagenesis_dataset = create_pyg_dataset("mutagenesis", "data/pyg/mutagenesis", ["type"], mutagenesis_feature_mappings, "k_hop_degrees")
    
    # rossmann_sample = sample_relational_distribution("rossmann-store-sales", 1000, seed=420)
    # mutagenesis_sample = sample_relational_distribution("mutagenesis", 10, seed=420)
    pass

if __name__ == "__main__":
    main()
