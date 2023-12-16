import os
import shutil
import random

import torch
import networkx as nx
from torch_geometric.utils import from_networkx
from torch_geometric.data import InMemoryDataset

from src.data.utils import load_metadata
from src.data_modelling.table_to_graph import database_to_graph, database_to_subgraphs
from src.data_modelling.feature_engineering import add_index, add_k_hop_degrees, filter_graph_features_with_mapping, add_k_hop_vectors

############################################################################################
DEFAULTS = {
    "rossmann-store-sales": {
        "features": ["node_type"],
        "feature_mappings": {"node_type": {"store": 0, "test": 1}},
        "target": "k_hop_vectors"
    },
    "mutagenesis": {
        "features": ["node_type"],
        "feature_mappings": {"node_type": {"molecule": 0, "atom": 1, "bond": 2}},
        "target": "k_hop_vectors"
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
        self.save(self.data_list, self.processed_paths[0])


############################################################################################


def sample_relational_distribution(dataset_name, num_graphs, seed=42):    
    subgraphs, _ = database_to_subgraphs(dataset_name)
    # set seed and sample with replacement
    random.seed(seed)
    sampled_subgraphs = random.choices(subgraphs, k = num_graphs)
    last_id = 0
    for i in range(len(sampled_subgraphs)):
        sampled_subgraphs[i] = nx.convert_node_labels_to_integers(sampled_subgraphs[i], first_label=last_id)
        last_id += len(sampled_subgraphs[i].nodes)
    
    return nx.compose_all(sampled_subgraphs)

############################################################################################


def pyg_dataset_from_graph(G, dataset_name, features=None, feature_mappings=None, target=None):
    # check if we need to set default parameters
    if features is None:
        features = DEFAULTS[dataset_name]["features"]
    if feature_mappings is None:
        feature_mappings = DEFAULTS[dataset_name]["feature_mappings"]
    if target is None:
        target = DEFAULTS[dataset_name]["target"]
    
    G = add_index(G)
    if target == "k_hop_degrees":
        G = add_k_hop_degrees(G, k=2)
        target_length = 2
    elif target == "k_hop_vectors":
        G = add_k_hop_vectors(G, k=3, node_types=load_metadata(dataset_name).get_tables())
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


def create_pyg_dataset(dataset_name, features=None, feature_mappings=None, target=None):
    # load the graph representing the database
    G, _ = database_to_graph(dataset_name)
    return pyg_dataset_from_graph(G, dataset_name, features=features, feature_mappings=feature_mappings, target=target)


############################################################################################

def main():    
    # whole dataset
    rossmann_dataset = create_pyg_dataset("rossmann-store-sales")
    mutagenesis_dataset = create_pyg_dataset("mutagenesis")
    # # sample
    rossmann_sample = sample_relational_distribution("rossmann-store-sales", 1000)
    mutagenesis_sample = sample_relational_distribution("mutagenesis",  1000)
    
    
    # k-hop vectors (k-hop degrees by type)
    # whole dataset
    rossmann_dataset = create_pyg_dataset("rossmann-store-sales", target="k_hop_vectors")
    mutagenesis_dataset = create_pyg_dataset("mutagenesis", target="k_hop_vectors")
    # sample
    rossmann_sample = sample_relational_distribution("rossmann-store-sales", 100)
    mutagenesis_sample = sample_relational_distribution("mutagenesis",  100)


if __name__ == "__main__":
    main()
