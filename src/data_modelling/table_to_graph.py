import pandas as pd
import networkx as nx

from src.data.utils import load_tables, load_metadata, get_root_table

###########################################################################################

def tables_to_graph(edge_index, source, target, source_attrs_df=None, target_attrs_df=None, directed=True):
        
    # generate graph from edge connections in the df
    create_using=nx.DiGraph() if directed else nx.Graph()
    G = nx.from_pandas_edgelist(edge_index, source=source, target=target, create_using=create_using)
    
    # add the atributes for source nodes
    if (source_attrs_df is not None) and (not source_attrs_df.empty):
        source_attrs_df = source_attrs_df.set_index(source)
        G.add_nodes_from(source_attrs_df.to_dict('index').items())
    
    # add the atributes for target nodes
    if (target_attrs_df is not None) and (not target_attrs_df.empty):
        target_attrs_df = target_attrs_df.set_index(target)
        G.add_nodes_from(target_attrs_df.to_dict('index').items())

    edge_types = {}
    for _, row in edge_index.iterrows():
        edge_types[(row[source], row[target])] = row['edge_type']
        

    nx.set_edge_attributes(G, edge_types, name='edge_type')
    
    return G


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


def database_to_graph(database_name, split="train", directed=True):
    """Convert a supported database to a graph.

    Args:
        database_name (str): name of the database. Currently supported: "rossmann-store-sales", "mutagenesis".
        split (str, optional): The data split to load. Defaults to "train".
    """
    
    # load the data and metadata
    tables = load_tables(database_name, split=split) # dict of dataframes
    metadata = load_metadata(database_name) # sdv metadata object (built from a metadata.json file)
    
    # initialize empty graph
    G = nx.DiGraph() if directed else nx.Graph()
    
    # starting id for primary key mappings
    starting_id = 0
    key_mappings = {}
    
    # loop over all of the tables
    for parent_table_name in metadata.get_tables():
        
        parent_pk = metadata.get_primary_key(parent_table_name)
        parent_table = tables[parent_table_name].copy()
        parent_table["node_type"] = parent_table_name
        if parent_table_name in key_mappings:
            parent_id_mapping = key_mappings[parent_table_name]
        else:
            parent_id_mapping = {parent_id: i + starting_id for i, parent_id in enumerate(parent_table[parent_pk])}
            key_mappings[parent_table_name] = parent_id_mapping
            starting_id += len(parent_id_mapping)    
        # remap the ids
        parent_table[parent_pk] = parent_table[parent_pk].map(parent_id_mapping)
        
        # loop over all of the children of the current table
        for child_table_name in metadata.get_children(parent_table_name):
            
            child_pk = metadata.get_primary_key(child_table_name)
            child_table = tables[child_table_name].copy()        
            child_table["node_type"] = child_table_name
            # create a mapping for parent and child ids (the primary keys are not necessarily integers so we remap them to integers)
            if child_table_name in key_mappings:
                child_id_mapping = key_mappings[child_table_name]
            else:
                child_id_mapping = {child_id: i + starting_id for i, child_id in enumerate(child_table[child_pk])}
                key_mappings[child_table_name] = child_id_mapping
                starting_id += len(child_id_mapping)
            # remap the ids
            child_table[child_pk] = child_table[child_pk].map(child_id_mapping)
            
            
            
            # loop over all of the foreign keys of the child table
            for foreign_key in metadata.get_foreign_keys(parent_table_name, child_table_name):
                # remap the foreign key of the child table
                child_table[foreign_key] = child_table[foreign_key].map(parent_id_mapping)
                
                # create the edge index used to build the graph
                edge_index = pd.DataFrame()
                edge_index[parent_pk] = child_table[foreign_key]
                edge_index[child_pk] = child_table[child_pk]
                edge_index["edge_type"] = foreign_key

                H = tables_to_graph(edge_index, source=parent_pk, target=child_pk, source_attrs_df=parent_table, target_attrs_df=child_table)
                G = nx.compose(G, H)
    
    root_node_ids = list(key_mappings[get_root_table(database_name)].values())
    return G, root_node_ids


def database_to_subgraphs(database_name, split="train", directed=True):
    # TOOD: write docstring
    G, G_root_node_ids = database_to_graph(database_name, split=split, directed=directed)
    return graph_to_subgraphs(G, G_root_node_ids), G_root_node_ids


def update_node_features(G, df, node_type, ids):
    df["node_type"] = node_type
    
    keys = G.nodes(data=True)[ids[0]].keys()
    node_attrs_dict = {}
    for node_id, (i, row) in zip(ids, df.iterrows()):
        new_attrs = {key: row[key] for key in keys}
        node_attrs_dict[node_id] = new_attrs
    
    nx.set_node_attributes(G, node_attrs_dict)
    return G


###########################################################################################


def main():
    G_rossmann, rossmann_root_node_ids = database_to_graph("rossmann-store-sales")
    rossmann_tables = load_tables("rossmann-store-sales", split="train")
    G_rossmann_updated = update_node_features(G_rossmann.copy(), rossmann_tables["store"], "store", ids=rossmann_root_node_ids)
    assert G_rossmann.nodes(data=True) == G_rossmann_updated.nodes(data=True), "The node features were not updated correctly"
    
    G_mutagenesis, mutagenesis_root_node_ids = database_to_graph("mutagenesis")
    mutagenesis_tables = load_tables("mutagenesis", split="train")
    G_mutagenesis_updated = update_node_features(G_mutagenesis.copy(), mutagenesis_tables["molecule"], "molecule", ids=mutagenesis_root_node_ids)
    assert G_mutagenesis.nodes(data=True) == G_mutagenesis_updated.nodes(data=True), "The node features were not updated correctly"


if __name__ == "__main__":  
    main()
