import os
import shutil
import random

import torch
import networkx as nx
from torch_geometric.utils import from_networkx
from torch_geometric.data import InMemoryDataset

from src.data_modelling.table_to_graph import database_to_graph, database_to_subgraphs
from src.data_modelling.feature_engineering import add_index, add_k_hop_degrees, filter_graph_features_with_mapping, add_k_hop_vectors

############################################################################################
DEFAULTS = {
    "rossmann-store-sales": {
        "features": ["type"],
        "feature_mappings": {"type": {"store": 0, "test": 1}},
        "target": "k_hop_degrees"
    },
    "mutagenesis": {
        "features": ["type"],
        "feature_mappings": {"type": {"molecule": 0, "atom": 1, "bond": 2}},
        "target": "k_hop_degrees"
    },
}


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
        # TODO: make sure we overwrite the existing data if we call process
        # if os.path.exists(self.root):
        #     shutil.rmtree(self.root)
        self.save(self.data_list, self.processed_paths[0])


############################################################################################


def sample_relational_distribution(dataset_name, num_graphs, features=None, feature_mappings=None, target=None, seed=42):    
    # check if we need to set default parameters
    if features is None:
        features = DEFAULTS[dataset_name]["features"]
    if feature_mappings is None:
        feature_mappings = DEFAULTS[dataset_name]["feature_mappings"]
    if target is None:
        target = DEFAULTS[dataset_name]["target"]
    
    subgraphs, _ = database_to_subgraphs(dataset_name)
    
    # add the index and target features
    if target == "k_hop_degrees":
        subgraphs = [add_k_hop_degrees(G, k=2) for G in subgraphs]
        target_length = 2
    elif target == "k_hop_vectors":
        subgraphs = [add_k_hop_vectors(G, k=3) for G in subgraphs]
        # target_length = subgraphs[0].nodes[0]['k_hop_vectors'].shape[1]
        target_length = len(subgraphs[0].nodes[0]['k_hop_vectors'])
    else:
        raise ValueError(f"Target {target} not supported")
    
    # set seed and sample with replacement
    random.seed(seed)
    sampled_subgraphs = random.choices(subgraphs, k = num_graphs)
    
    # filter and map the features
    features_to_keep = [*features, target]
    sampled_subgraphs = [filter_graph_features_with_mapping(G, features_to_keep, feature_mappings) for G in sampled_subgraphs]

    # convert to pytorch geometric Data objects
    sampled_subgraphs = [from_networkx(G, group_node_attrs=features_to_keep) for G in sampled_subgraphs]
    
    previous_graph_nodes = 0
    for i in range(len(sampled_subgraphs)):
        sampled_subgraphs[i].edge_index += previous_graph_nodes
        previous_graph_nodes += len(sampled_subgraphs[i].x)
    # concat all of the tensors
    all_graphs_data = sampled_subgraphs[0]
    for data in sampled_subgraphs[1:]:
        all_graphs_data.edge_index = torch.cat((all_graphs_data.edge_index, data.edge_index), dim=1)
        all_graphs_data.x = torch.cat((all_graphs_data.x, data.x), dim=0)
    # set the index
    all_graphs_data.index = torch.tensor(range(all_graphs_data.x.shape[0]))
    # separate the features and target
    all_graphs_data.y = all_graphs_data.x[:, (all_graphs_data.x.shape[1] - target_length):].type(torch.float)
    all_graphs_data.x = all_graphs_data.x[:, :(all_graphs_data.x.shape[1] - target_length)].type(torch.float)
    
    if os.path.exists(f"data/pyg/{dataset_name}/sample"):
        shutil.rmtree(f"data/pyg/{dataset_name}/sample")
    dataset = GraphRelationalDataset(root=f"data/pyg/{dataset_name}/sample", data_list=[all_graphs_data])
    dataset.process()
    return dataset


############################################################################################


def create_pyg_dataset(dataset_name, features=None, feature_mappings=None, target=None):
    # check if we need to set default parameters
    if features is None:
        features = DEFAULTS[dataset_name]["features"]
    if feature_mappings is None:
        feature_mappings = DEFAULTS[dataset_name]["feature_mappings"]
    if target is None:
        target = DEFAULTS[dataset_name]["target"]
    
    
    # load the graph representing the database
    G, _ = database_to_graph(dataset_name)
    
    G = add_index(G)
    if target == "k_hop_degrees":
        G = add_k_hop_degrees(G, k=2)
        target_length = 2
    elif target == "k_hop_vectors":
        G = add_k_hop_vectors(G, k=3)
        # target_length = G.nodes[0]['k_hop_vectors'].shape[1]
        target_length = len(G.nodes[0]['k_hop_vectors'])
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
    
    if os.path.exists(f"data/pyg/{dataset_name}"):
        shutil.rmtree(f"data/pyg/{dataset_name}")
    dataset = GraphRelationalDataset(root=f"data/pyg/{dataset_name}", data_list=[data])
    dataset.process()
    return dataset


############################################################################################

def main():    
    rossmann_dataset = create_pyg_dataset("rossmann-store-sales")
    mutagenesis_dataset = create_pyg_dataset("mutagenesis")
    
    rossmann_sample = sample_relational_distribution("rossmann-store-sales", 1000)
    mutagenesis_sample = sample_relational_distribution("mutagenesis",  1000)


if __name__ == "__main__":
    main()
