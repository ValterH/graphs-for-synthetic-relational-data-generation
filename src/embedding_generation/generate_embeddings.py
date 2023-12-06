import pathlib
import json
import torch
import torch.nn as nn
from torch_geometric.utils import from_networkx

# pyg implemenation of GIN (need to specify the model architecture when instantiating)
from torch_geometric.nn.models import GIN
# custom GIN with 2 layers
from src.embedding_generation.GINModel import GINModel

from src.data_modelling.table_to_graph import (
    get_rossmann_graph,
    get_rossmann_subgraphs,
    get_mutagenesis_graph,
    get_mutagenesis_subgraphs,
    graph_to_subgraphs,
    filter_graph_features_with_mapping
)
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


# TODO: how should we actually train
# take the whole graph for now
data = mutagenesis_dataset[0]
############################################################################################


model = GINModel(num_features=1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_func = nn.MSELoss()
epochs = 200

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = loss_func(out, data.y)
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")


model.eval()
with torch.no_grad():
    embeddings = model(data.x, data.edge_index, return_embeds=True)
    
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