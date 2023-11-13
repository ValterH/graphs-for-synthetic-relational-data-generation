import pathlib
import numpy as np
import pandas as pd
import networkx as nx
from torch_geometric.utils.convert import to_networkx, from_networkx
from torch_geometric.data import Dataset, HeteroData

###########################################################################################

FILE_ABS_PATH = pathlib.Path(__file__) # absolute path of this file

###########################################################################################

# TODO: think about this. Do we really want weekly connected components?
# def mutagenesis_to_nx_components(dir_path):
#     G = mutagenesis_to_nx_graph(dir_path)
#     Gs = []
#     for weekly_connected_component in nx.weakly_connected_components(G):
#         # create a subgraph for each connected component
#         Gs.append(G.subgraph(weekly_connected_component))
#     return Gs

def mutagenesis_to_nx_components(dir_path, parent_nodes):
    G = mutagenesis_to_nx_graph(dir_path)
    Gs = []
    for parent_node in parent_nodes:
        # TODO: 1 graph for each node in the parent table since we're doing 1:N
        pass
    return Gs


def mutagenesis_to_nx_graph(dir_path):
    # paths of the csv files
    dir_path = pathlib.Path(dir_path)
    atom_path = dir_path / "atom.csv"
    bond_path = dir_path / "bond.csv"
    molecule_path = dir_path / "molecule.csv"
    
    molecule_df = pd.read_csv(molecule_path, index_col=None)
    molecule_df["Molecule"] = molecule_df.index
    molecule_mapping = {molecule: id for id, molecule in zip(molecule_df["molecule_id"], molecule_df["Molecule"])}
    
    atom_df = pd.read_csv(atom_path, index_col=None)
    atom_df["Atom"] = atom_df.index + molecule_df["Molecule"].max() + 1
    atom_mapping = {atom: id for id, atom in zip(atom_df["atom_id"], atom_df["Atom"])}
    
    bond_df = pd.read_csv(bond_path, index_col=None)
    bond_df["Bond"] = bond_df.index + atom_df["Atom"].max() + 1
    
    
    # connect molecules to atoms (1:N)
    atom_df["Molecule"] = atom_df["molecule_id"].map(molecule_mapping)
    G_molecules_atoms = nx.from_pandas_edgelist(atom_df, source="Molecule", target="Atom", edge_attr=None, create_using=nx.DiGraph)
    
    # connect atoms to bonds (1:N)
    bond_df["Atom_1"] = bond_df["atom1_id"].map(atom_mapping)
    bond_df["Atom_2"] = bond_df["atom2_id"].map(atom_mapping)
    G_atoms_bonds_1 = nx.from_pandas_edgelist(bond_df, source="Atom_1", target="Bond", edge_attr=None, create_using=nx.DiGraph)
    G_atoms_bonds_2 = nx.from_pandas_edgelist(bond_df, source="Atom_2", target="Bond", edge_attr=None, create_using=nx.DiGraph)
    # merge the two graphs
    G_atoms_bonds = nx.compose(G_atoms_bonds_1, G_atoms_bonds_2)
    
    # TODO: what to do with molecule:bond 1:N relationship?
    
    # merge graphs
    G = nx.compose(G_molecules_atoms, G_atoms_bonds)
    
    
    # TODO: add node attributes
    
    
    return G

###########################################################################################

# TODO: wrap in a Dataset class
class MutagenesisDataset(Dataset):
    pass

###########################################################################################

def main():
    dir_path = FILE_ABS_PATH.parent.parent.parent / "data" / "mutagenesis"
    Gs = mutagenesis_to_nx_components(dir_path)


if __name__ == "__main__":
    main()
