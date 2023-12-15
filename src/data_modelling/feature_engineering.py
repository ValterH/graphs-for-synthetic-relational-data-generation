import numpy as np
import pandas as pd
import networkx as nx

from src.data_modelling.table_to_graph import database_to_graph, database_to_subgraphs


############################################################################################

def get_neighbors_np(G, nodes):
    """
    Returns a numpy array of unique neighbors for a given list of nodes in a graph G.

    Parameters:
    G (networkx.Graph): The graph
    nodes (list, int, or numpy array): A list of nodes, a single node, or a numpy array of nodes in the graph

    Returns:
    numpy array: An array of unique neighbors of the given nodes
    """
    # Ensure nodes is a numpy array
    if not isinstance(nodes, np.ndarray):
        nodes = np.array(nodes, ndmin=1)

    neighbors = set()
    for node in nodes:
        # Add neighbors of the current node to the set
        neighbors.update(nx.neighbors(G, node))
    
    return np.array(list(neighbors))


def get_k_hop_neighbors_np(G, nodes, k):
    """
    Returns a numpy array of k-hop neighbors for a given node or list of nodes in a graph G.

    Parameters:
    G (networkx.Graph): The graph
    nodes (list, int, or numpy array): A list of nodes, a single node, or a numpy array of nodes in the graph
    k (int): The number of hops

    Returns:
    numpy array: An array of k-hop neighbors of the given nodes
    """
    # Ensure nodes is a numpy array
    if not isinstance(nodes, np.ndarray):
        nodes = np.array(nodes, ndmin=1)

    if k == 0:
        return nodes
    else:
        return get_k_hop_neighbors_np(G, get_neighbors_np(G, nodes), k-1)


def get_node_type_counts_np(G, nodes, node_types):
    """
    Returns a numpy array where each index represents the count of nodes of a particular type in the list of nodes,
    starting indexing at 1 (no nodes of type 0).

    Parameters:
    G (networkx.Graph): The graph
    nodes (list, int, or numpy array): A list of nodes, a single node, or a numpy array of nodes in the graph
    num_types (int): The total number of different node types, starting from 1

    Returns:
    numpy array: An array with counts of each node type
    """
    # Ensure nodes is a numpy array
    if not isinstance(nodes, np.ndarray):
        nodes = np.array(nodes, ndmin=1)

    # Initialize the count array with zeros
    type_counts = {node_type: 0 for node_type in node_types}

    # Increment the count for each node's type
    for node in nodes:
        if node in G:
            node_type = G.nodes[node]['type']
            if node_type not in type_counts:
                type_counts[node_type] = 0
            type_counts[node_type] += 1

    # Convert the dictionary to a numpy array
    type_counts = np.array(list(type_counts.values()))
    return type_counts


def get_k_hop_node_type_matrix_np(G, node, k, node_types, include_self=False, flatten=False):
    """
    Returns a numpy matrix where each row represents the count array of node types in the k-th hop from the original node,
    optionally including the type count of the node itself.

    Parameters:
    G (networkx.Graph): The graph
    node (int): The original node
    k (int): The number of hops
    num_types (int): The total number of different node types, starting from 1
    include_self (bool): If True, includes the type count of the node itself in the matrix
    flatten (bool): If True, flattens the matrix into a single array

    Returns:
    numpy matrix or array: A matrix with rows representing k-hop node type counts, optionally flattened
    """
    # Initialize the matrix
    type_matrix = []

    # If include_self is True, add the type count of the node itself
    if include_self:
        type_matrix.append(get_node_type_counts_np(G, [node], node_types))

    # Get k-hop neighbors for each level
    current_level_nodes = np.array([node])
    for _ in range(k):
        next_level_nodes = np.array([], dtype=int)
        for n in current_level_nodes:
            next_level_nodes = np.union1d(next_level_nodes, get_neighbors_np(G, n))
        # Get type counts for the current level and add to the matrix
        type_matrix.append(get_node_type_counts_np(G, next_level_nodes, node_types))
        # Prepare for the next level
        current_level_nodes = next_level_nodes

    type_matrix = np.array(type_matrix)

    return type_matrix.flatten() if flatten else type_matrix


    
def get_k_hop_matrix_dict(G, k, node_types, include_self=False, flatten=True):
    """
    Returns a dictionary of k-hop node type matrices for each node in the graph.

    Parameters:
    G (networkx.Graph): The graph

    Returns:
    dict: A dictionary of k-hop node type matrices for each node in the graph
    """
    k_hop_matrix_dict = {}
    for node in G.nodes:
        k_hop_matrix_dict[node] = get_k_hop_node_type_matrix_np(G, node, k=k, node_types = node_types, include_self=include_self, flatten=flatten)

    return k_hop_matrix_dict


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

def add_k_hop_vectors(G, k, undirected=True, include_self=False):

    if undirected:
        G = G.copy().to_undirected()
    # TODO: write docstring
    node_types = {G.nodes[node]['type'] for node in G.nodes}
    all_k_hop_vectors_dict = get_k_hop_matrix_dict(G, k, node_types, include_self=include_self, flatten=True)
    nx.set_node_attributes(G, all_k_hop_vectors_dict, "k_hop_vectors")
    return G


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
    G_rossmann = add_k_hop_vectors(G_rossmann, k=2, include_self=True)
    G_rossmann = filter_graph_features_with_mapping(G_rossmann, ["index", "type", "k_hop_degrees"], {"type": {"store": 0, "sale": 1}})
    
    G_mutagenesis, _ = database_to_graph("mutagenesis")
    G_mutagenesis = add_k_hop_degrees(G_mutagenesis, k=2)
    G_mutagenesis = add_index(G_mutagenesis)
    G_mutagenesis = add_k_hop_vectors(G_mutagenesis, k=2, include_self=True)
    G_mutagenesis = filter_graph_features_with_mapping(G_mutagenesis, ["index", "type", "k_hop_degrees"], {"type": {"molecule": 0, "atom": 1, "bond": 2}})


if __name__ == "__main__":
    main()
