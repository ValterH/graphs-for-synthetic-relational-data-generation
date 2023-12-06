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




# def convert_networkx_to_pyg(graph):
#     # Convert NetworkX graph to PyG Data object
#     data = from_networkx(graph)

#     # Extract node features 'y' from NetworkX graph and convert to tensor
#     features = []
#     node_id_mapping = {}  # Mapping of node IDs to their indices
#     for i, (node_id, node_data) in enumerate(graph.nodes(data=True)):
#         features.append([node_data['y']])
#         node_id_mapping[node_id] = i  # Map the original node ID to its index
#     data.x = torch.tensor(features, dtype=torch.float)

#     # Calculate and add node degrees
#     degrees = [val for _, val in graph.degree()]
#     data.y = torch.tensor(degrees, dtype=torch.float).view(-1, 1)

#     return data, node_id_mapping

# G1, G1_roots = mutagenesis_to_graph_1("data/mutagenesis") # NOTE: relative to root of the project


############################################################################################

rossmann_dataset = get_rossmann_dataset()
rossmann_subgraph_dataset = get_rossmann_subgraphs_dataset()

mutagenesis_dataset = get_mutagenesis_dataset()
mutagenesis_subgraph_dataset = get_mutagenesis_subgraphs_dataset()

pass
############################################################################################


# # get data from our graph
# data, node_mapping = convert_networkx_to_pyg(G1)

# # Assuming each node has one feature
# model = GINModel(num_features=1)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
# loss_func = nn.MSELoss()
# epochs = 200


# # Training loop
# for epoch in range(epochs):
#     model.train()
#     optimizer.zero_grad()
#     out = model(data.x, data.edge_index)
#     loss = loss_func(out, data.y)
#     loss.backward()
#     optimizer.step()

#     if epoch % 10 == 0:
#         print(f"Epoch {epoch}, Loss: {loss.item()}")


# # Extract embeddings after the final epoch
# model.eval()
# with torch.no_grad():
#     embeddings = model(data.x, data.edge_index, return_embeds=True)


# root_embeddings = {}
# for node_id in G1:
#     node_index = node_mapping[node_id]
#     embedding = embeddings[node_index]
#     root_embeddings[node_id] = embedding


# # Convert tensor embeddings to lists for JSON serialization
# root_embeddings_serializable = {node_id: embedding.tolist() for node_id, embedding in root_embeddings.items()}


# # Define the file path
# file_path = "src/embedding_generation/root_embeddings.json"

# # Save to a JSON file
# with open(file_path, 'w') as f:
#     json.dump(root_embeddings_serializable, f)

# print(f"Saved embeddings to {file_path}")