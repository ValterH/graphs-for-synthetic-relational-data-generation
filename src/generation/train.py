import os
import json

import torch
import numpy as np
import pandas as pd

from diffusion import train_diff
from autoencoder import train_vae
from utils_train import preprocess
from tabsyn.latent_utils import get_input_train
from src.data.utils import load_tables, load_metadata
from src.data_modelling.pyg_datasets import create_pyg_dataset
from src.embedding_generation.generate_embeddings import train_gin, generate_embeddings

############################################################################################

def train_pipline(dataset_name, run, retrain_vae=False, cond="mlp", epochs_gnn=250, epochs_vae=4000, epochs_diff=4000, seed=42):
    torch.manual_seed(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # read data
    tables = load_tables(dataset_name, split='train')
    metadata = load_metadata(dataset_name)
    # create graph
    dataset = create_pyg_dataset(dataset_name)
    
    # train GIN and compute structural embeddings using GIN
    print('Training GIN')
    gin_model_save_path = f"models/gin_embeddings/{dataset_name}/{run}/"
    train_gin(dataset_name, gin_model_save_path, epochs=epochs_gnn)
    
    # generate GIN embeddings
    gin_data_save_path = f"data/gin_embeddings/{dataset_name}/{run}/"
    generate_embeddings(dataset, metadata, model_path=gin_model_save_path, data_save_path=gin_data_save_path)
    
    
    # train generative model for each table (VAE with latent conditional diffusion)
    for table in metadata.get_tables():
        # train vae
        
        if retrain_vae or not os.path.exists(f'ckpt/{table}/vae/decoder.pt'):
            print(f'Training VAE for table {table}')
            X_num, X_cat, categories, d_numerical = preprocess(f'tabsyn/data/{table}', task_type = 'binclass', concat=False)
            train_vae(X_num, X_cat, categories, d_numerical, ckpt_dir = f'ckpt/{table}/vae' , epochs=epochs_vae, device=device)
        else:
            print(f'Reusing VAE for table {table}')
            
        # combine vae embeddings from parent tables with structural embeddings from the current table to obtain condition diffusion
        gin_embeddings  =  pd.read_csv(f'{gin_data_save_path}{table}_embeddings.csv', index_col=0)
        gin_embeddings  =  gin_embeddings.to_numpy()
        conditional_embeddings = [gin_embeddings]
        for parent in metadata.get_parents(table):
            parent_embeddings = []
            # parent embeddings are stored in the same order as the parent table
            parent_latent_embeddings = np.load(f'ckpt/{parent}/vae/train_z.npy')
            parent_ids = tables[parent][metadata.get_primary_key(parent)].tolist()
            for fk in metadata.get_foreign_keys(parent, table):
                fks = tables[table][fk]
                cond_embeddings = np.zeros((len(fks), parent_latent_embeddings.shape[1], parent_latent_embeddings.shape[2]))
                for i, id in enumerate(fks):
                    parent_id = parent_ids.index(id)
                    cond_embeddings[i] = parent_latent_embeddings[parent_id]
                parent_embeddings.append(cond_embeddings)
            # average embeddings from different parents
            parent_embeddings = np.mean(parent_embeddings, axis=0)
            parent_embeddings = parent_embeddings.reshape((parent_embeddings.shape[0], -1))
            conditional_embeddings.append(parent_embeddings)
        if len(conditional_embeddings) > 1:
            conditional_embeddings = np.concatenate(conditional_embeddings, axis=1)
        else:
            conditional_embeddings = conditional_embeddings[0]
        
        os.makedirs(f'ckpt/{table}/{run}', exist_ok=True)
        np.save(f'ckpt/{table}/{run}/cond_train_z.npy', conditional_embeddings)

        # train conditional diffusion
        train_z, train_z_cond, _, ckpt_path, _ = get_input_train(table, is_cond=True, run=run)
        print(f'Training conditional diffusion for table {table}')
        train_diff(train_z, train_z_cond, ckpt_path, epochs=epochs_diff, is_cond=True, cond=cond, device=device)


############################################################################################

def main():
    # TODO: remove hardcoded arguments and call for both datasets
    
    dataset_name = 'rossmann-store-sales'
    retrain_vae = False
    cond = 'mlp'
    run = 'TEST'
    epochs_vae, epochs_diff = 2, 2
    
    train_pipline(dataset_name=dataset_name, run=run, retrain_vae=retrain_vae, cond=cond, epochs_vae=epochs_vae, epochs_diff=epochs_diff)


if __name__ == '__main__':
    main()