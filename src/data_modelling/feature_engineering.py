import numpy as np
import pandas as pd
import networkx as nx

from src.data_modelling.table_to_graph import database_to_graph, database_to_subgraphs


############################################################################################

# TODO: k_hop_degrees based on node type
def k_hop_degrees_node_type(G, node, k):
    pass


############################################################################################


def all_nodes_k_hop_degrees(G, k, undirected=True):
    """Get the k-hop degrees of all nodes in a graph.

    Args:
        G (nx.Graph): The graph.
        k (int): The k-hop degree.
        undirected (bool, optional): Whether to consider the graph as undirected. Defaults to True.

    Returns:
        dict: The k-hop degrees of all nodes in the graph.
    """
    
    if undirected:
        G = G.copy().to_undirected()
    
    all_k_hop_degrees_dict = {}
    for node in G.nodes():
        all_k_hop_degrees_dict[node] = k_hop_degrees(G, node, k)
        
    return all_k_hop_degrees_dict


def k_hop_degrees(G, node, k):
    """Get the k-hop degrees of a node in a graph.

    Args:
        G (nx.Graph): The graph.
        node (int): The node id.
        k (int): The k-hop degree.

    Returns:
        list: The k-hop degrees of the node.
    """
    
    k_hop_neighbors = nx.single_source_shortest_path_length(G, node, cutoff=k)
    k_hop_degrees = [list(k_hop_neighbors.values()).count(i) for i in range(1, k + 1)]
    return k_hop_degrees


############################################################################################


def add_k_hop_degrees(G, k, undirected=True):
    # TODO: write docstring
    
    all_k_hop_degrees_dict = all_nodes_k_hop_degrees(G, k, undirected=undirected)
    nx.set_node_attributes(G, all_k_hop_degrees_dict, "k_hop_degrees")
    return G

def add_index(G):
    # TODO: write docstring
    
    id_dict = {id: id for id in G.nodes()}
    nx.set_node_attributes(G, id_dict, "index")
    return G


############################################################################################


def filter_graph_features_with_mapping(G, features_to_keep, feature_mappings):
    # TODO: write docstring
    
    filtered_graph = nx.Graph()

    for node, data in G.nodes(data=True):
        updated_features = {}
        for feature in features_to_keep:
            # Check if the feature needs mapping
            if feature in feature_mappings:
                # Map the feature value using the specified mapping
                mapping = feature_mappings[feature]
                updated_features[feature] = mapping.get(data.get(feature))
            else:
                # Keep the feature as is
                updated_features[feature] = data.get(feature)

        # Add a new node to the filtered graph with filtered features
        filtered_graph.add_node(node, **updated_features)

    # keep the connections between the nodes as they were
    filtered_graph.add_edges_from(G.edges())
    
    return filtered_graph


############################################################################################


def main():
    G_rossmann, _ = database_to_graph("rossmann-store-sales")
    G_rossmann = add_k_hop_degrees(G_rossmann, k=2)
    G_rossmann = add_index(G_rossmann)
    G_rossmann = filter_graph_features_with_mapping(G_rossmann, ["index", "type", "k_hop_degrees"], {"type": {"store": 0, "sale": 1}})
    
    G_mutagenesis, _ = database_to_graph("mutagenesis")
    G_mutagenesis = add_k_hop_degrees(G_mutagenesis, k=2)
    G_mutagenesis = add_index(G_mutagenesis)
    G_mutagenesis = filter_graph_features_with_mapping(G_mutagenesis, ["index", "type", "k_hop_degrees"], {"type": {"molecule": 0, "atom": 1, "bond": 2}})


if __name__ == "__main__":
    main()
