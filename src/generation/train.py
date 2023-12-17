import os
import argparse

import torch
import numpy as np

from src.data.utils import load_tables, load_metadata
from src.generation.diffusion import train_diff
from src.generation.autoencoder import train_vae
from src.generation.utils_train import preprocess
from src.generation.tabsyn.latent_utils import get_input_train
from src.data_modelling.table_to_graph import database_to_graph
from src.embedding_generation.hetero_gnn import train_hetero_gnn
from src.data_modelling.pyg_datasets import pyg_dataset_from_graph
from src.conditioning import simple_message_passing, gnn_message_passing
from src.embedding_generation.gin_embeddings import train_gin, generate_embeddings


############################################################################################

def train_pipline(dataset_name, run, retrain_vae=False, cond="mlp", message_passing='simple', k=10, epochs_gnn=250, epochs_vae=4000, epochs_diff=4000, seed=42):
    torch.manual_seed(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # read data
    tables = load_tables(dataset_name, split='train')
    metadata = load_metadata(dataset_name)
    
    # create graph 
    G, _ = database_to_graph(dataset_name)
    
    if message_passing == 'simple':
        dataset = pyg_dataset_from_graph(G, dataset_name)
        # train GIN and compute structural embeddings using GIN
        print('Training GIN')
        gin_model_save_path = f"models/gin_embeddings/{dataset_name}/{run}/"
        train_gin(dataset_name, gin_model_save_path, epochs=epochs_gnn)
    
        # generate GIN embeddings
        gin_data_save_path = f"data/gin_embeddings/{dataset_name}/{run}/"
        generate_embeddings(dataset, metadata, model_path=gin_model_save_path, data_save_path=gin_data_save_path)
    elif message_passing == 'gnn':
        masked_tables = metadata.get_tables()
    
    
    # train generative model for each table (VAE with latent conditional diffusion)
    for table in metadata.get_tables():
        # train vae
        
        if retrain_vae or not os.path.exists(f'ckpt/{table}/vae/decoder.pt'):
            print(f'Training VAE for table {table}')
            X_num, X_cat, categories, d_numerical = preprocess(f'tabsyn/data/{table}', concat=False)
            train_vae(X_num, X_cat, categories, d_numerical, ckpt_dir = f'ckpt/{table}/vae' , epochs=epochs_vae, device=device)
        else:
            print(f'Reusing VAE for table {table}')
            
        # combine vae embeddings from parent tables with structural embeddings from the current table to obtain condition diffusion
        if message_passing == 'simple':
            conditional_embeddings, _, _, _ = simple_message_passing(metadata, tables, table, gin_data_save_path, train=True)
        elif message_passing == 'gnn':
            train_hetero_gnn(dataset_name, table, masked_tables)
            conditional_embeddings, _, _, _ = gnn_message_passing(metadata, G, table, masked_tables, dataset_name, k=k)
            masked_tables.remove(table)
        else:
            raise NotImplementedError(f'Message passing method {message_passing} not implemented')
        
        os.makedirs(f'ckpt/{table}/{run}', exist_ok=True)
        np.save(f'ckpt/{table}/{run}/cond_train_z.npy', conditional_embeddings)

        # train conditional diffusion
        train_z, train_z_cond, _, ckpt_path, _ = get_input_train(table, is_cond=True, run=run)
        print(f'Training conditional diffusion for table {table}')
        train_diff(train_z, train_z_cond, ckpt_path, epochs=epochs_diff, is_cond=True, cond=cond, device=device)


############################################################################################


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-name', type=str, required=True)
    parser.add_argument('--cond', type=str, default='mlp')
    parser.add_argument('--message-passing', type=str, default='gnn')
    parser.add_argument('--denoising-steps', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--retrain_vae', action='store_true')
    parser.add_argument('--epochs-vae', type=int, default=4000)
    parser.add_argument('--epochs-diff', type=int, default=10000)
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_name = args.dataset_name
    cond = args.cond
    message_passing = args.message_passing
    run = f"{cond}_{message_passing}"
    retrain_vae = args.retrain_vae
    epochs_vae = args.epochs_vae
    epochs_diff = args.epochs_diff
    
    train_pipline(dataset_name=dataset_name, run=run, retrain_vae=retrain_vae, cond=cond, message_passing=message_passing, epochs_vae=epochs_vae, epochs_diff=epochs_diff)


if __name__ == '__main__':
    main()