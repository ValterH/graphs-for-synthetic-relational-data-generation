import pathlib
import numpy as np
import pandas as pd
import networkx as nx
from torch_geometric.utils.convert import to_networkx, from_networkx
from torch_geometric.data import Dataset, HeteroData

###########################################################################################

FILE_ABS_PATH = pathlib.Path(__file__) # absolute path of this file

###########################################################################################


def mutagenesis_to_nx_components(G, parent_nodes):
    Gs = []
    for parent_node in parent_nodes:
        reachable_nodes = nx.descendants(G, parent_node).add(parent_node)
        Gs.append(G.subgraph(reachable_nodes))
    return Gs


def mutagenesis_to_nx_graph(dir_path):
    # paths of the csv files
    dir_path = pathlib.Path(dir_path)
    atom_path = dir_path / "atom.csv"
    bond_path = dir_path / "bond.csv"
    molecule_path = dir_path / "molecule.csv"
    
    molecule_df = pd.read_csv(molecule_path, index_col=None)
    molecule_df["Molecule"] = molecule_df.index
    molecule_mapping = {id: molecule for id, molecule in zip(molecule_df["molecule_id"], molecule_df["Molecule"])}
    
    atom_df = pd.read_csv(atom_path, index_col=None)
    atom_df["Atom"] = atom_df.index + molecule_df["Molecule"].max() + 1
    atom_mapping = {id: atom for id, atom in zip(atom_df["atom_id"], atom_df["Atom"])}
    
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
    
    
    # get each component of the graph
    Gs = mutagenesis_to_nx_components(G, molecule_df["Molecule"])
    
    return G, Gs

###########################################################################################

# TODO: wrap in a Dataset class
class MutagenesisDataset(Dataset):
    pass

###########################################################################################

def main():
    dir_path = FILE_ABS_PATH.parent.parent.parent / "data" / "mutagenesis"
    G, Gs = mutagenesis_to_nx_graph(dir_path)
    pass


if __name__ == "__main__":
    main()
