from torch_geometric.utils.sparse import to_edge_index

from src.data_modelling.deepsnap_datasets import deepsnap_dataset_from_graph
from src.embedding_generation.hetero_gnn import compute_hetero_gnn_embeddings


def gnn_message_passing(metadata, G, target_table, masked_tables, dataset_name, k=10):
    # combine vae embeddings from parent tables with structural embeddings from the current table to obtain condition diffusion
    hetero_graph = deepsnap_dataset_from_graph(G, metadata, masked_tables, label_encoders_path=f'data/hetero_graph/{dataset_name}_{target_table}_{k}_label_encoders.pkl')
    ids = hetero_graph.node_label_index[target_table].tolist()
    original_ids = hetero_graph.node_to_graph_mapping[target_table].tolist()
    conditional_embeddings = compute_hetero_gnn_embeddings(hetero_graph, dataset_name, target_table)
    foreign_keys = dict()
    for parent in metadata.get_parents(target_table):
        for fk in metadata.get_foreign_keys(parent, target_table):     
            sparse_fk_edge_index = hetero_graph.edge_index[(parent, fk, target_table)]
            fk_edge_index, _ = to_edge_index(sparse_fk_edge_index)
            # get parent ids from edge index
            assert fk_edge_index[0].tolist() == ids
            foreign_keys[fk] = fk_edge_index[1].tolist()

    return conditional_embeddings, ids, original_ids, foreign_keys