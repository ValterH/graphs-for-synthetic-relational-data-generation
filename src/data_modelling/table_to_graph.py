import pathlib
import pandas as pd
import networkx as nx

###########################################################################################
SEED = 42
FILE_ABS_PATH = pathlib.Path(__file__) # absolute path of this file

rossman_dir_path = FILE_ABS_PATH.parent.parent.parent / "data" / "rossmann"
mutagenesis_dir_path = FILE_ABS_PATH.parent.parent.parent / "data" / "mutagenesis"

###########################################################################################


# get a dataframe with source and target columns
# the columns contain the id's of the nodes in the graph
def tables_to_graph(df, source, target, source_attrs_df=None, target_attrs_df=None, directed=True):
    
    # generate graph from edge connections in the df
    create_using=nx.DiGraph() if directed else nx.Graph()
    G = nx.from_pandas_edgelist(df, source=source, target=target, create_using=create_using)
    
    # add the atributes for source nodes
    if (source_attrs_df is not None) and (not source_attrs_df.empty):
        source_attrs_df = source_attrs_df.set_index(source)
        G.add_nodes_from(source_attrs_df.to_dict('index').items())
    
    # add the atributes for target nodes
    if (target_attrs_df is not None) and (not target_attrs_df.empty):
        target_attrs_df = target_attrs_df.set_index(target)
        G.add_nodes_from(target_attrs_df.to_dict('index').items())
    
    return G


def rossman_to_graph(dir_path=rossman_dir_path, train=True):
    # read in the data
    store_df = pd.read_csv(pathlib.Path(dir_path) / "store.csv")
    sales_df = pd.read_csv(pathlib.Path(dir_path) / "train.csv") if train else pd.read_csv(pathlib.Path(dir_path) / "test.csv")
    
    # each store and sale has a unique id
    store_id_mapping = {store_id: i for i, store_id in enumerate(store_df["Store"].unique())}
    sales_id_mapping = {sales_id: i + len(store_id_mapping) for i, sales_id in enumerate(sales_df.index)}
    
    # features for stores and sales
    # TODO: create dataframes with encoded features that are useful
    store_attrs_df = store_df
    store_attrs_df["Store"] = store_df["Store"].map(store_id_mapping)
    store_attrs_df["y"] = "store"
    
    sales_attrs_df = sales_df
    sales_attrs_df["Sale"] = sales_df.index.map(sales_id_mapping)
    sales_attrs_df = sales_attrs_df.drop(columns=["Store"])
    sales_attrs_df["y"] = "sale"
    
    # edges between stores and sales
    store_sales_df = pd.DataFrame()
    store_sales_df["Store"] = sales_df["Store"].map(store_id_mapping)
    store_sales_df["Sale"] = sales_df.index.map(sales_id_mapping)
    
    # root nodes (parent table)
    root_nodes = store_sales_df["Store"].unique().tolist()
    
    # generate the graph
    G = tables_to_graph(store_sales_df, source="Store", target="Sale", source_attrs_df=store_attrs_df, target_attrs_df=sales_attrs_df)

    # return the graph and the root nodes
    return G, root_nodes


def mutagenesis_to_graph(dir_path=mutagenesis_dir_path):
    # read in the data
    molecule_df = pd.read_csv(pathlib.Path(dir_path) / "molecule.csv")
    atom_df = pd.read_csv(pathlib.Path(dir_path) / "atom.csv")
    bond_df = pd.read_csv(pathlib.Path(dir_path) / "bond.csv")
    
    # each molecule, atom and bond has a unique id
    molecule_id_mapping = {molecule_id: i for i, molecule_id in enumerate(molecule_df["molecule_id"].unique())}
    atom_id_mapping = {atom_id: i + len(molecule_id_mapping) for i, atom_id in enumerate(atom_df["atom_id"].unique())}
    bond_id_mapping = {bond_id: i + len(molecule_id_mapping) + len(atom_id_mapping) for i, bond_id in enumerate(bond_df.index)}
    
    # features for stores and sales
    # TODO: create dataframes with encoded features that are useful
    molecule_attrs_df = molecule_df
    molecule_attrs_df["Molecule"] = molecule_df["molecule_id"].map(molecule_id_mapping)
    molecule_attrs_df = molecule_attrs_df.drop(columns=["molecule_id"])
    molecule_attrs_df["y"] = "molecule"
    
    atom_attrs_df = atom_df
    atom_attrs_df["Atom"] = atom_df["atom_id"].map(atom_id_mapping)
    atom_attrs_df = atom_attrs_df.drop(columns=["molecule_id", "atom_id"])
    atom_attrs_df["y"] = "atom"
    
    bond_attrs_df = bond_df
    bond_attrs_df["Bond"] = bond_df.index.map(bond_id_mapping)
    bond_attrs_df = bond_attrs_df.drop(columns=["atom1_id", "atom2_id"])
    bond_attrs_df["y"] = "bond"
    
    
    # first bipartite component
    molecule_atom_df = pd.DataFrame()
    molecule_atom_df["Molecule"] = atom_df["molecule_id"].map(molecule_id_mapping)
    molecule_atom_df["Atom"] = atom_df["atom_id"].map(atom_id_mapping)
    G_molecule_to_atom = tables_to_graph(molecule_atom_df, source="Molecule", target="Atom", source_attrs_df=molecule_attrs_df, target_attrs_df=atom_attrs_df)

    # second bipartite component
    atom_bond_df = pd.DataFrame()
    atom_bond_df["Atom1"] = bond_df["atom1_id"].map(atom_id_mapping)
    atom_bond_df["Atom2"] = bond_df["atom2_id"].map(atom_id_mapping)
    atom_bond_df["Bond"] = bond_df.index.map(bond_id_mapping)
    # connect atom1 and atom2 to bond
    atom_attrs_df = atom_attrs_df.rename(columns={"Atom": "Atom1"})
    G_atom1_to_bond = tables_to_graph(atom_bond_df, source="Atom1", target="Bond", source_attrs_df=atom_attrs_df, target_attrs_df=bond_attrs_df)
    atom_attrs_df = atom_attrs_df.rename(columns={"Atom1": "Atom2"})
    G_atom2_to_bond = tables_to_graph(atom_bond_df, source="Atom2", target="Bond", source_attrs_df=atom_attrs_df, target_attrs_df=bond_attrs_df)
    atom_attrs_df = atom_attrs_df.rename(columns={"Atom2": "Atom"})
    # combine the two graphs
    G_atom_to_bond = nx.compose(G_atom1_to_bond, G_atom2_to_bond)

    # root nodes (parent table)
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


def filter_graph_features_with_mapping(graph, features_to_keep, feature_mapping):
    filtered_graph = nx.Graph()

    for node, data in graph.nodes(data=True):
        updated_features = {}
        for feature in features_to_keep:
            # Check if the feature needs mapping
            if feature in feature_mapping:
                # Map the feature value using the specified mapping
                mapping = feature_mapping[feature]
                updated_features[feature] = mapping.get(data.get(feature))
            else:
                # Keep the feature as is
                updated_features[feature] = data.get(feature)

        # Add a new node to the filtered graph with filtered features
        filtered_graph.add_node(node, **updated_features)

    # keep the connections between the nodes as they were
    filtered_graph.add_edges_from(graph.edges())
    
    return filtered_graph


###########################################################################################

def get_rossmann_graph(rossmann_dir_path=rossman_dir_path, root_nodes=True, features=None, feature_mappings=None):
    G_rossmann, rossmann_root_nodes = rossman_to_graph(rossmann_dir_path)
    
    # filter features
    if features is not None:
        G_rossmann = filter_graph_features_with_mapping(G_rossmann, features, feature_mappings)
    
    # also return the root nodes
    if root_nodes:
        return G_rossmann, rossmann_root_nodes
    
    return G_rossmann


def get_rossmann_subgraphs(rossmann_dir_path=rossman_dir_path, features=None, feature_mappings=None):
    if features is None:
        G_rossmann, rossmann_root_nodes = get_rossmann_graph(rossmann_dir_path, features=features, feature_mappings=feature_mappings)
    else:
        G_rossmann, rossmann_root_nodes = get_rossmann_graph(rossmann_dir_path, features=features, feature_mappings=feature_mappings)
    
    return graph_to_subgraphs(G_rossmann, rossmann_root_nodes)


def get_mutagenesis_graph(mutagenesis_dir_path=mutagenesis_dir_path, root_nodes=True, features=None, feature_mappings=None):
    G_mutagenesis, mutagenesis_root_nodes = mutagenesis_to_graph(mutagenesis_dir_path)
    
    # filter features
    if features is not None:
        G_mutagenesis = filter_graph_features_with_mapping(G_mutagenesis, features, feature_mappings)
    
    # also return the root nodes
    if root_nodes:
        return G_mutagenesis, mutagenesis_root_nodes
    
    return G_mutagenesis


def get_mutagenesis_subgraphs(mutagenesis_dir_path=mutagenesis_dir_path, features=None, feature_mappings=None):
    if features is None:
        G_mutagenesis, mutagenesis_root_nodes = get_mutagenesis_graph(mutagenesis_dir_path, features=features, feature_mappings=feature_mappings)
    else:
        G_mutagenesis, mutagenesis_root_nodes = get_mutagenesis_graph(mutagenesis_dir_path, features=features, feature_mappings=feature_mappings)
    
    return graph_to_subgraphs(G_mutagenesis, mutagenesis_root_nodes)

###########################################################################################

def main():
    pass

if __name__ == "__main__":  
    main()
