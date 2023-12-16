import os

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch_geometric.nn.models import GIN
from torch_geometric.utils import get_embeddings

from src.data.utils import load_metadata
from src.data_modelling.pyg_datasets import create_pyg_dataset

############################################################################################

GIN_DEFAULTS = {
    # model params
    "hidden_channels": 32,
    "jk": "last",
    "norm": "batch",
    
    # optimizer params
    "lr": 0.01,
    
    # training params
    "epochs": 250,

    # data params
    "target": "k_hop_vectors",
}


############################################################################################

def train_gin(dataset_name, model_save_path, target=None, hidden_channels=32, jk="last", norm="batch", lr=0.01, epochs=250, seed=42):
    if target is None:
        target = GIN_DEFAULTS["target"]
    torch.manual_seed(seed)
    
    
    metadata = load_metadata(dataset_name)
    dataset = create_pyg_dataset(dataset_name, target=target)
    # our dataset contains only a single graph (union of disjoint graphs) representing the entire database
    data = dataset[0] 
    
    # model
    feature_dim = data.x.shape[1]
    num_tables = len(metadata.get_tables())
    target_dim = data.y.shape[1]
    model = GIN(in_channels=feature_dim, hidden_channels=hidden_channels, num_layers=(num_tables - 1), out_channels=target_dim, jk=jk, norm=norm)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_func = nn.MSELoss()
    
    # training
    pbar = tqdm(range(epochs))
    for epoch in pbar:
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = loss_func(out, data.y)
        loss.backward()
        optimizer.step()
        
        pbar.set_description(f"Training {dataset_name} GIN model (Epoch: {epoch} | Loss: {loss.item():.4f})")
    
    # save model
    os.makedirs(model_save_path, exist_ok=True)
    torch.save(model.state_dict(), model_save_path + "model.pt")


def get_gin_embeddings(model, data):
    model.eval()
    with torch.no_grad():
        embeddings = get_embeddings(model, data.x, data.edge_index)
        last_layer_embeddings = embeddings[-1]
    return last_layer_embeddings


############################################################################################


def generate_embeddings(dataset, metadata, model_path, data_save_path, hidden_channels=32, jk="last", norm="batch"):
    # data
    data = dataset[0]
    
    # model
    feature_dim = data.x.shape[1]
    num_tables = len(metadata.get_tables())
    target_dim = data.y.shape[1]
    model = GIN(in_channels=feature_dim, hidden_channels=hidden_channels, num_layers=(num_tables - 1), out_channels=target_dim, jk=jk, norm=norm)
    # load trained model
    model.load_state_dict(torch.load(model_path + "model.pt"))
    
    # get embeddings
    last_layer_embeddings = get_gin_embeddings(model, data)

    table_mapping = {table_name: table_id for table_id, table_name in enumerate(metadata.get_tables())}
    # write generated embeddings to a dataframe
    for table_name, table_id in table_mapping.items():
        # select the embeddings for this table
        table_mask = (data.x[:, 0] == table_id)
        node_ids = data.index[table_mask]
        table_embeddings_df = pd.DataFrame(last_layer_embeddings[table_mask].numpy(), index=node_ids.numpy()).sort_index()
        
        os.makedirs(data_save_path, exist_ok=True)
        table_embeddings_df.to_csv(data_save_path + f"{table_name}_embeddings.csv")
        
        for parent in metadata.get_parents(table_name):
            for i, fk in enumerate(metadata.get_foreign_keys(parent, table_name)):
                
                pyg_ids = np.where(table_mask)[0]
                parent_pyg_ids = np.where(data.x[:, 0] == table_mapping[parent])[0]

                edge_index = pd.DataFrame(data.edge_index.T, columns=['source', 'target'])
                
                # select only edges that contain this table's nodes as targets 
                # (currently we have this stored as undirected graph in the edge index so this is not the case)
                edge_index = edge_index[edge_index['target'].isin(pyg_ids)]
                edge_index = edge_index[edge_index['source'].isin(parent_pyg_ids)]
                
                # select the ith edge for each parent node (eg. atom1_id , atom2_id)
                edge_index = edge_index[edge_index.groupby("target").cumcount() == i]
                
                # parent = parent node id
                fks = pd.DataFrame(data.index[edge_index['source'].values], columns=['parent'])
                
                fks.index = node_ids.numpy()
                # id = child node id
                fks['id'] = node_ids.numpy()
                fks.sort_index(inplace=True)
                fks.to_csv(data_save_path + f"{table_name}_{fk}_fks.csv", index=False)


############################################################################################


def main():
    train_gin("rossmann-store-sales", "models/gin_embeddings/rossmann-store-sales/", target="k_hop_degrees")
    train_gin("mutagenesis", "models/gin_embeddings/mutagenesis/", target="k_hop_vectors")
    
    generate_embeddings(create_pyg_dataset("rossmann-store-sales", target="k_hop_degrees"), load_metadata("rossmann-store-sales"), "models/gin_embeddings/rossmann-store-sales/", "data/gin_embeddings/rossmann-store-sales/")
    generate_embeddings(create_pyg_dataset("mutagenesis", target="k_hop_vectors"), load_metadata("mutagenesis"), "models/gin_embeddings/mutagenesis/", "data/gin_embeddings/mutagenesis/")

    # sampled dataset embeddings
    from src.data_modelling.pyg_datasets import sample_relational_distribution, pyg_dataset_from_graph

    G_rossmann = sample_relational_distribution("rossmann-store-sales", 200)
    dataset_rossmann = pyg_dataset_from_graph(G_rossmann, "rossmann-store-sales", target="k_hop_degrees")
    generate_embeddings(dataset_rossmann, load_metadata("rossmann-store-sales"), "models/gin_embeddings/rossmann-store-sales/", "data/gin_embeddings/rossmann-store-sales/generated/")


    G_mutagenesis = sample_relational_distribution("mutagenesis", 200)
    dataset_mutagenesis = pyg_dataset_from_graph(G_mutagenesis, "mutagenesis", target="k_hop_vectors")
    generate_embeddings(dataset_mutagenesis, load_metadata("mutagenesis"), "models/gin_embeddings/mutagenesis/", "data/gin_embeddings/mutagenesis/generated/")


if __name__ == "__main__":
    main()
