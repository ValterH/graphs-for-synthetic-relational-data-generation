import pathlib
import numpy as np
import pandas as pd
import networkx as nx

###########################################################################################
SEED = 42
FILE_ABS_PATH = pathlib.Path(__file__) # absolute path of this file
###########################################################################################

# TODO:
# 1. add features to all the nodes
# 2. add the option to return the dataset as one big graph or a list of disconnected graphs
# 3. ...


# get a dataframe with source and target columns
# the columns contain the id's of the nodes in the graph
def tables_to_graph(df, source, target, directed=True):
    
    # generate a graph from the dataframe
    create_using=nx.DiGraph() if directed else nx.Graph()
    G = nx.from_pandas_edgelist(df, source=source, target=target, create_using=create_using)
    
    # TODO: features (if the df is a join of the two tables we can do it by specifying the source and target columns we want for features)
    # TODO: disconnected graphs (figure out if we want strongly/weakly connected graphs or just all the reachable nodes from the source node in the parent table)
    
    return G


# read in rossmann data and convert it to a graph
def rossman_to_graph(dir_path, train=True):
    store_df = pd.read_csv(dir_path / "store.csv")
    sales_df = pd.read_csv(dir_path / "train.csv") if train else pd.read_csv(dir_path / "test.csv")
    
    store_id_mapping = {store_id: i for i, store_id in enumerate(store_df["Store"].unique())}
    sales_id_mapping = {sales_id: i + len(store_id_mapping) for i, sales_id in enumerate(sales_df.index)}
    
    store_sales_df = pd.DataFrame()
    store_sales_df["Store"] = sales_df["Store"].map(store_id_mapping)
    store_sales_df["Sale"] = sales_df.index.map(sales_id_mapping)
    
    root_nodes = store_sales_df["Store"].unique().tolist()
    G = tables_to_graph(store_sales_df, source="Store", target="Sale")

    return G, root_nodes


# read in mutagenesis data and convert it to a graph
def mutagenesis_to_graph(dir_path):
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
            G_molecule_to_atom.nodes[node]['y'] = 'molecule'
        else:
            G_molecule_to_atom.nodes[node]['y'] = 'atom'

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
            G_atom_to_bond.nodes[node]['y'] = 'atom'
        else:
            G_atom_to_bond.nodes[node]['y'] = 'bond'
    
    
    root_nodes = molecule_atom_df["Molecule"].unique().tolist()
    # combine the two bipartite components
    G = nx.compose(G_molecule_to_atom, G_atom_to_bond)

    return G, root_nodes


# assume we have a parent table with 1:N relationship
# -> we can split the graph by choosing the nodes in the parent table as root nodes and generating a tree for each root node
# NOTE: we can imagine this as modeling the dataset tables as a list of connected rows
def graph_to_subgraphs(G, root_nodes):
    subgraphs = []
    for root_node in root_nodes:
        subgraph_nodes = nx.descendants(G, root_node)
        subgraph_nodes.add(root_node) # add for sets works inplace
        subgraphs.append(G.subgraph(subgraph_nodes))
    return subgraphs


###########################################################################################


###########################################################################################
rossman_dir_path = FILE_ABS_PATH.parent.parent.parent / "data" / "rossmann"
mutagenesis_dir_path = FILE_ABS_PATH.parent.parent.parent / "data" / "mutagenesis"

ROSSMANN_GRAPH, ROSSMAN_ROOT_NODES = rossman_to_graph(rossman_dir_path)
MUTAGENESIS_GRAPH, MUTAGENESIS_ROOT_NODES = mutagenesis_to_graph(mutagenesis_dir_path)
###########################################################################################

def main():
    pass

if __name__ == "__main__":  
    main()
