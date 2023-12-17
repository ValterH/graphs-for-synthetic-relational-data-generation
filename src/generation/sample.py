import os
import argparse

import torch
import numpy as np

from src.generation.diffusion import sample_diff
from src.data.utils import load_metadata, save_tables
from src.generation.autoencoder import create_latent_embeddings
from src.data_modelling.table_to_graph import update_node_features
from src.embedding_generation.gin_embeddings import generate_embeddings
from src.conditioning import simple_message_passing, gnn_message_passing
from src.data_modelling.pyg_datasets import sample_relational_distribution, pyg_dataset_from_graph


############################################################################################

def sample(dataset_name, num_samples, run, cond="mlp", message_passing='simple', k=10, seed=None, denoising_steps=50):
    if seed is not None:
        torch.manual_seed(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # read data
    metadata = load_metadata(dataset_name)
    tables = dict()
    
    # create graph
    G = sample_relational_distribution(dataset_name, num_samples)
    
    if message_passing == 'simple':
        dataset = pyg_dataset_from_graph(G, dataset_name)
        # compute structural embeddings using GIN
        model_path = f"models/gin_embeddings/{dataset_name}/{run}/"
        embeddings_save_path = f"data/gin_embeddings/{dataset_name}/{run}/generated/"
        generate_embeddings(dataset, metadata, model_path=model_path, data_save_path=embeddings_save_path)
    elif message_passing == 'gnn':
        masked_tables = metadata.get_tables()
    
    
    # for each table in dataset
    ids = {}
    for table in metadata.get_tables():
        if message_passing == 'simple':
            conditional_embeddings, ids[table], original_ids, foreign_keys = simple_message_passing(metadata, tables, table, embeddings_save_path)
        elif message_passing == 'gnn':
            conditional_embeddings, ids[table], original_ids, foreign_keys = gnn_message_passing(metadata, G, table, masked_tables, dataset_name, k=k)
            masked_tables.remove(table)
        else:
            raise NotImplementedError(f'Message passing method {message_passing} not implemented')
        
        os.makedirs(f'ckpt/{table}/{run}/gen', exist_ok=True)
        np.save(f'ckpt/{table}/{run}/gen/cond_z.npy', conditional_embeddings)
        
        # sample diffusion
        df = sample_diff(table, run, is_cond=True, cond=cond, device=device, foreign_keys=foreign_keys, ids=ids[table], denoising_steps=denoising_steps)
        tables[table] = df
        print(f'Successfully sampled data for table {table}')
        print(df.head())

        if message_passing == 'simple' and metadata.get_children(table):
            create_latent_embeddings(df, table, device = device)
        elif message_passing == 'gnn':
            G = update_node_features(G, df, node_type=table, ids=original_ids)
        
    
    save_tables(tables, dataset_name, data_type=f'synthetic/ours/{run}')


############################################################################################

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-name', type=str, required=True)
    parser.add_argument('--num-samples', type=int, required=True)
    parser.add_argument('--cond', type=str, default='mlp')
    parser.add_argument('--message-passing', type=str, default='gnn')
    parser.add_argument('--denoising-steps', type=int, default=100)
    parser.add_argument('--seed', type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args() 
    dataset_name = args.dataset_name
    num_samples = args.num_samples
    cond = args.cond
    message_passing = args.message_passing
    run = f"{cond}_{message_passing}"
    denoising_steps = args.denoising_steps
    
    sample(dataset_name=dataset_name, num_samples=num_samples, run=run, cond=cond, message_passing=message_passing, denoising_steps=denoising_steps)


if __name__ == '__main__':
    main()
