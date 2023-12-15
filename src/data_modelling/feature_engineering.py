import numpy as np
import pandas as pd
import networkx as nx

from src.data_modelling.table_to_graph import database_to_graph


############################################################################################

def all_nodes_k_hop_vectors(G, k, node_types, undirected=True):
    
    if undirected:
        G = G.copy().to_undirected()
    
    # NOTE: if we get a graph which doesn't contain all of the node types, we will get an incorrect result - so we can't infer the node types from the graph
    # node_types = nx.get_node_attributes(G, "type")
    # node_types = {node_type for node_type in node_types.values()}
    # # convert to list and sort by alphabetical order to ensure consistent ordering
    # node_types = sorted(list(node_types))
    
    all_k_hop_vectors_dict = {}
    for node in G.nodes():
        all_k_hop_vectors_dict[node] = k_hop_vectors(G, node, k, node_types)
    
    return all_k_hop_vectors_dict


def k_hop_vectors(G, node, k, node_types):
    
    k_hop_neighbors_distance = nx.single_source_shortest_path_length(G, node, cutoff=k)
    k_hop_neighbour_types = np.array([G.nodes(data=True)[node]["type"] for node in list(k_hop_neighbors_distance.keys())])
    
    k_hop_vector = []
    for node_type in node_types:
        
        mask = (k_hop_neighbour_types == node_type)
        distances_masked = np.array(list(k_hop_neighbors_distance.values()))[mask]
        k_hop_vector_type = [np.count_nonzero(distances_masked == i) for i in range(1, k + 1)]
        
        k_hop_vector.extend(k_hop_vector_type)

    return k_hop_vector


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

def add_k_hop_vectors(G, k, node_types, undirected=True):
    all_k_hop_vectors_dict = all_nodes_k_hop_vectors(G, k, node_types, undirected=undirected)
    nx.set_node_attributes(G, all_k_hop_vectors_dict, "k_hop_vectors")
    return G


def add_k_hop_degrees(G, k, undirected=True):
    all_k_hop_degrees_dict = all_nodes_k_hop_degrees(G, k, undirected=undirected)
    nx.set_node_attributes(G, all_k_hop_degrees_dict, "k_hop_degrees")
    return G


def add_index(G):
    id_dict = {id: id for id in G.nodes()}
    nx.set_node_attributes(G, id_dict, "index")
    return G


############################################################################################


def filter_graph_features_with_mapping(G, features_to_keep, feature_mappings):
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
    G_rossmann = add_index(G_rossmann)
    G_rossmann = add_k_hop_degrees(G_rossmann, k=2)
    G_rossmann = add_k_hop_vectors(G_rossmann, k=2, node_types=["store", "test"])
    G_rossmann = filter_graph_features_with_mapping(G_rossmann, ["index", "type", "k_hop_degrees", "k_hop_vectors"], {"type": {"store": 0, "sale": 1}})
    
    G_mutagenesis, _ = database_to_graph("mutagenesis")
    G_mutagenesis = add_index(G_mutagenesis)
    G_mutagenesis = add_k_hop_degrees(G_mutagenesis, k=2)
    G_mutagenesis = add_k_hop_vectors(G_mutagenesis, k=2, node_types=["molecule", "atom", "bond"])
    G_mutagenesis = filter_graph_features_with_mapping(G_mutagenesis, ["index", "type", "k_hop_degrees", "k_hop_vectors"], {"type": {"molecule": 0, "atom": 1, "bond": 2}})


if __name__ == "__main__":
    main()
