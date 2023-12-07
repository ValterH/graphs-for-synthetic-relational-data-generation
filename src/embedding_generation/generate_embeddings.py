import pathlib
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
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

############################################################################################

# rossmann_dataset = get_rossmann_dataset()
# rossmann_subgraph_dataset = get_rossmann_subgraphs_dataset()

mutagenesis_dataset = get_mutagenesis_dataset()
# mutagenesis_subgraph_dataset = get_mutagenesis_subgraphs_dataset()

pass

# TODO: how should we actually train
# take the whole graph for now
data = mutagenesis_dataset[0]
############################################################################################


# model = GINModel(num_features=1)

model = GIN(in_channels=1, hidden_channels=32, num_layers=2, out_channels=1, jk="last")

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_func = nn.MSELoss()
# epochs = 200
epochs = 500

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = loss_func(out, data.y)
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")


# model.eval()
# with torch.no_grad():
#     embeddings = model(data.x, data.edge_index, return_embeds=True)


model.eval()
with torch.no_grad():
    embeddings = get_embeddings(model, data.x, data.edge_index)
    last_layer_embeddings = embeddings[-1]
pass

moluecule_embeddings = last_layer_embeddings[data.x[:, 0] == 0]
atom_embeddings = last_layer_embeddings[data.x[:, 0] == 1]
bond_embeddings = last_layer_embeddings[data.x[:, 0] == 2]

pass

############################################################################################

# # Convert tensor embeddings to lists for JSON serialization
# root_embeddings_serializable = {node_id: embedding.tolist() for node_id, embedding in root_embeddings.items()}


# # Define the file path
# file_path = "src/embedding_generation/root_embeddings.json"

# # Save to a JSON file
# with open(file_path, 'w') as f:
#     json.dump(root_embeddings_serializable, f)

# print(f"Saved embeddings to {file_path}")

############################################################################################


# Convert embeddings to a numpy array
embeddings_array = last_layer_embeddings.cpu().numpy()

# Perform PCA
pca = PCA(n_components=2)  # You can adjust the number of components as needed
embeddings_pca = pca.fit_transform(embeddings_array)

# Visualization
plt.figure(figsize=(8, 6))

pca_0_x, pca_0_y = embeddings_pca[data.x[:, 0] == 0, 0], embeddings_pca[data.x[:, 0] == 0, 1]
pca_1_x, pca_1_y = embeddings_pca[data.x[:, 0] == 1, 0], embeddings_pca[data.x[:, 0] == 1, 1]
pca_2_x, pca_2_y = embeddings_pca[data.x[:, 0] == 2, 0], embeddings_pca[data.x[:, 0] == 2, 1]

plt.scatter(pca_0_x, pca_0_y, alpha=0.5, label="Molecule", color="red")
plt.scatter(pca_1_x, pca_1_y, alpha=0.5, label="Atom", color="blue")
plt.scatter(pca_2_x, pca_2_y, alpha=0.5, label="Bond", color="green")

# plt.scatter(embeddings_pca[:, 0], embeddings_pca[:, 1], alpha=0.5)
plt.title('PCA Visualization of Embeddings')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

pass