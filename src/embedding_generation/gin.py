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


# read in mutagenesis data and convert it to a graph
def mutagenesis_to_graph_1(dir_path):
    molecule_df = pd.read_csv(dir_path + "/molecule.csv")
    atom_df = pd.read_csv(dir_path + "/atom.csv")
    bond_df = pd.read_csv(dir_path + "/bond.csv")
    
    molecule_id_mapping = {molecule_id: i for i, molecule_id in enumerate(molecule_df["molecule_id"].unique())}
    atom_id_mapping = {atom_id: i + len(molecule_id_mapping) for i, atom_id in enumerate(atom_df["atom_id"].unique())}
    bond_id_mapping = {bond_id: i + len(molecule_id_mapping) + len(atom_id_mapping) for i, bond_id in enumerate(bond_df.index)}
    
    # first bipartite component
    molecule_atom_df = pd.DataFrame()
    molecule_atom_df["Molecule"] = atom_df["molecule_id"].map(molecule_id_mapping)
    molecule_atom_df["Atom"] = atom_df["atom_id"].map(atom_id_mapping)
    G_molecule_to_atom = tables_to_graph(molecule_atom_df, source="Molecule", target="Atom")

    # Label molecules and atoms in G_molecule_to_atom
    for node in G_molecule_to_atom.nodes():
        if node in molecule_id_mapping.values():
            G_molecule_to_atom.nodes[node]['y'] = 0
        else:
            G_molecule_to_atom.nodes[node]['y'] = 1

    # second bipartite component
    atom_bond_df = pd.DataFrame()
    atom_bond_df["Atom1"] = bond_df["atom1_id"].map(atom_id_mapping)
    atom_bond_df["Atom2"] = bond_df["atom2_id"].map(atom_id_mapping)
    atom_bond_df["Bond"] = bond_df.index.map(bond_id_mapping)
    # connect atom1 to bond
    G_atom1_to_bond = tables_to_graph(atom_bond_df, source="Atom1", target="Bond")
    # connect atom2 to bond
    G_atom2_to_bond = tables_to_graph(atom_bond_df, source="Atom2", target="Bond")
    # combine the two graphs
    G_atom_to_bond = nx.compose(G_atom1_to_bond, G_atom2_to_bond)

    # Label atoms and bonds in G_atom_to_bond
    for node in G_atom_to_bond.nodes():
        if node in atom_id_mapping.values():
            G_atom_to_bond.nodes[node]['y'] = 1
        else:
            G_atom_to_bond.nodes[node]['y'] = 2
    
    
    root_nodes = molecule_atom_df["Molecule"].unique().tolist()
    # combine the two bipartite components
    G = nx.compose(G_molecule_to_atom, G_atom_to_bond)

    return G, root_nodes

