import pandas as pd
import networkx as nx
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv

from src.data_modelling.table_to_graph import tables_to_graph


class GINModel(nn.Module):
    def __init__(self, num_features):
        super(GINModel, self).__init__()
        nn1 = nn.Sequential(nn.Linear(num_features, 32), nn.ReLU(), nn.Linear(32, 32))
        nn2 = nn.Sequential(nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 32))

        self.conv1 = GINConv(nn1)
        self.conv2 = GINConv(nn2)
        self.fc = nn.Linear(32, 1)  # Output layer for degree prediction

    def forward(self, x, edge_index, return_embeds=False):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))

        if return_embeds:
            return x  # Return embeddings after the last GINConv layer

        return self.fc(x)  # Continue to the final output layer
