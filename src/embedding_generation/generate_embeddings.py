import os
import pathlib
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch_geometric.utils import from_networkx, get_embeddings


# pyg implemenation of GIN (need to specify the model architecture when instantiating)
from torch_geometric.nn.models import GIN
# custom GIN with 2 layers
from src.embedding_generation.GINModel import GINModel
from src.data_modelling.pyg_datasets import (
    get_rossmann_dataset,
    get_rossmann_subgraphs_dataset,
    get_mutagenesis_dataset,
    get_mutagenesis_subgraphs_dataset
)

# Set a seed for PyTorch
SEED = 42  # Replace 42 with the desired seed value
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

############################################################################################

# train with a single graph represented in a Data object
def train(model, data, optimizer, loss_func, epochs=100):
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = loss_func(out, data.y)
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")


# TODO: should we train in batches with multiple disjoint graphs?
# train over a dataset with multiple graphs
# def train_batched(model, dataset, optimizer, loss_func, epochs=100, batch_size=32):
#     data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
#     for epoch in range(epochs):
#         model.train()
#         for batch in data_loader:
#             optimizer.zero_grad()
#             out = model(batch.x, batch.edge_index)
#             loss = loss_func(out, batch.y)
#             loss.backward()
#             optimizer.step()
        
#         if epoch % 10 == 0:
#             print(f"Epoch {epoch}, Loss: {loss.item()}")


# TODO: evaluate the model if we decide to evaluate separate stages
# def test():
#     pass


def get_gin_embeddings(model, data):
    model.eval()
    with torch.no_grad():
        embeddings = get_embeddings(model, data.x, data.edge_index)
        last_layer_embeddings = embeddings[-1]
    return last_layer_embeddings


############################################################################################


def get_rossmann_embeddings(model_save_path = "models/gin_embeddings/rossmann-store-sales/", data_save_path="data/gin_embeddings/rossmann-store-sales/"):
    
    # data
    rossmann_dataset = get_rossmann_dataset()
    data = rossmann_dataset[0]
    
    # model
    # model = GINModel(num_features=1)
    model = GIN(in_channels=1, hidden_channels=32, num_layers=1, out_channels=1, jk="last")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_func = nn.MSELoss()
    epochs = 200
    
    # training
    train(model, data, optimizer, loss_func, epochs=epochs)
    # save model
    os.makedirs(model_save_path, exist_ok=True)
    torch.save(model.state_dict(), model_save_path + "model.pt")
    
    # get embeddings
    last_layer_embeddings = get_gin_embeddings(model, data)

    # write generated embeddings to a dataframe
    stores_mask, sales_mask = data.x[:, 0] == 0, data.x[:, 0] == 1
    stores_embeddings_df = pd.DataFrame(last_layer_embeddings[stores_mask].numpy(), index=data.index[stores_mask].numpy()).sort_index()
    sales_embeddings_df = pd.DataFrame(last_layer_embeddings[sales_mask].numpy(), index=data.index[sales_mask].numpy()).sort_index()
    
    os.makedirs(data_save_path, exist_ok=True)
    stores_embeddings_df.to_csv(data_save_path + "store_embeddings.csv", index=False)
    sales_embeddings_df.to_csv(data_save_path + "test_embeddings.csv", index=False)


def get_mutagenesis_embeddings(model_save_path = "models/gin_embeddings/mutagenesis/", data_save_path="data/gin_embeddings/mutagenesis/"):
    
    # data
    mutagenesis_dataset = get_mutagenesis_dataset()
    data = mutagenesis_dataset[0]
    
    # model
    # model = GINModel(num_features=1)
    model = GIN(in_channels=1, hidden_channels=32, num_layers=2, out_channels=1, jk="last")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_func = nn.MSELoss()
    epochs = 200
    
    # training
    train(model, data, optimizer, loss_func, epochs=epochs)
    # save model
    os.makedirs(model_save_path, exist_ok=True)
    torch.save(model.state_dict(), model_save_path + "model.pt")
    
    # get embeddings
    last_layer_embeddings = get_gin_embeddings(model, data)

    # write generated embeddings to a dataframe
    molecule_mask, atom_mask, bond_mask = data.x[:, 0] == 0, data.x[:, 0] == 1, data.x[:, 0] == 2
    molecule_embeddings_df = pd.DataFrame(last_layer_embeddings[molecule_mask].numpy(), index=data.index[molecule_mask].numpy()).sort_index()
    atom_embeddings_df = pd.DataFrame(last_layer_embeddings[atom_mask].numpy(), index=data.index[atom_mask].numpy()).sort_index()
    bond_embeddings_df = pd.DataFrame(last_layer_embeddings[bond_mask].numpy(), index=data.index[bond_mask].numpy()).sort_index()
    
    os.makedirs(data_save_path, exist_ok=True)
    molecule_embeddings_df.to_csv(data_save_path + "molecule_embeddings.csv")
    atom_embeddings_df.to_csv(data_save_path + "atom_embeddings.csv")
    bond_embeddings_df.to_csv(data_save_path + "bond_embeddings.csv")


def generate_embeddings(dataset, metadata, table_mapping, model_save_path, data_save_path, in_channels=1, hidden_channels=32, num_layers=1, out_channels=1, jk="last"):
    # data
    data = dataset[0]
    
    # model
    # model = GINModel(num_features=1)
    model = GIN(in_channels=in_channels, hidden_channels=hidden_channels, num_layers=num_layers, out_channels=out_channels, jk=jk)
    # load model
    model.load_state_dict(torch.load(model_save_path + "model.pt"))
    
    # get embeddings
    last_layer_embeddings = get_gin_embeddings(model, data)

    # write generated embeddings to a dataframe
    for table_name, table_id in table_mapping.items():
        table_mask = data.x[:, 0] == table_id
        table_embeddings_df = pd.DataFrame(last_layer_embeddings[table_mask].numpy(), index=data.index[table_mask].numpy()).sort_index()
        os.makedirs(data_save_path, exist_ok=True)
        table_embeddings_df.to_csv(data_save_path + f"{table_name}_embeddings.csv", index=False)
        if metadata.get_parents(table_name):
            # TODO: this works only for rossmann we should add edge types to graphs
            node_ids = data.index[table_mask].numpy()
            edge_index = pd.DataFrame(data.edge_index.T, columns=['source', 'target'])
            fks = pd.DataFrame(data.edge_index[:, :len(table_mask)][0, table_mask], index=data.index[table_mask].numpy()).sort_index()
            fks.reset_index(inplace=True, drop = True)
            fks.to_csv(data_save_path + f"{table_name}_fks.csv", index=False)


        
############################################################################################


# # Convert embeddings to a numpy array
# embeddings_array = last_layer_embeddings.cpu().numpy()

# # Perform PCA
# pca = PCA(n_components=2)  # You can adjust the number of components as needed
# embeddings_pca = pca.fit_transform(embeddings_array)

# # Visualization
# plt.figure(figsize=(8, 6))

# pca_0_x, pca_0_y = embeddings_pca[data.x[:, 0] == 0, 0], embeddings_pca[data.x[:, 0] == 0, 1]
# pca_1_x, pca_1_y = embeddings_pca[data.x[:, 0] == 1, 0], embeddings_pca[data.x[:, 0] == 1, 1]
# pca_2_x, pca_2_y = embeddings_pca[data.x[:, 0] == 2, 0], embeddings_pca[data.x[:, 0] == 2, 1]

# plt.scatter(pca_0_x, pca_0_y, alpha=0.5, label="Molecule", color="red")
# plt.scatter(pca_1_x, pca_1_y, alpha=0.5, label="Atom", color="blue")
# plt.scatter(pca_2_x, pca_2_y, alpha=0.5, label="Bond", color="green")

# # plt.scatter(embeddings_pca[:, 0], embeddings_pca[:, 1], alpha=0.5)
# plt.title('PCA Visualization of Embeddings')
# plt.xlabel('Principal Component 1')
# plt.ylabel('Principal Component 2')
# plt.show()

############################################################################################


def main():
    rossmann_embeddings = get_rossmann_embeddings()
    mutagenesis_embeddings = get_mutagenesis_embeddings()
    
    pass


if __name__ == "__main__":
    main()
