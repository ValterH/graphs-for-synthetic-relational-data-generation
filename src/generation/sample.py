import os

import torch
import numpy as np
import pandas as pd

from diffusion import sample_diff
from autoencoder import create_latent_embeddings
from src.data.utils import load_metadata, save_tables
from src.data_modelling.pyg_datasets import sample_relational_distribution
from src.embedding_generation.generate_embeddings import generate_embeddings

############################################################################################

def sample(dataset_name, num_samples, run, cond="mlp", seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # read data
    metadata = load_metadata(dataset_name)
    tables = dict()
    
    # create graph
    dataset = sample_relational_distribution(dataset_name, num_samples)
    
    # compute structural embeddings using GIN
    model_path = f"models/gin_embeddings/{dataset_name}/{run}/"
    embeddings_save_path = f"data/gin_embeddings/{dataset_name}/{run}/generated/"
    generate_embeddings(dataset, metadata, model_path=model_path, data_save_path=embeddings_save_path)
    
    
    # for each table in dataset
    ids = {}
    for table in metadata.get_tables():
        gin_embeddings = pd.read_csv(f'{embeddings_save_path}{table}_embeddings.csv', index_col=0)
        ids[table] = gin_embeddings.index.tolist()
        gin_embeddings = gin_embeddings.to_numpy()
        
        foreign_keys = dict()
        conditional_embeddings = [gin_embeddings]
        for parent in metadata.get_parents(table):
            parent_embeddings = []
            
            # parent embeddings are stored in the same order as the parent table
            parent_latent_embeddings = np.load(f'ckpt/{parent}/vae/gen_z.npy')
            parent_ids = tables[parent][metadata.get_primary_key(parent)].tolist()
            for fk in metadata.get_foreign_keys(parent, table):     
                fks = pd.read_csv(f'{embeddings_save_path}{table}_{fk}_fks.csv')
                foreign_keys[fk] = fks['parent'].tolist()
                cond_embeddings = np.zeros((len(fks), parent_latent_embeddings.shape[1], parent_latent_embeddings.shape[2]))
                for i, id in enumerate(foreign_keys[fk]):
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

        os.makedirs(f'ckpt/{table}/{run}/gen', exist_ok=True)
        np.save(f'ckpt/{table}/{run}/gen/cond_z.npy', conditional_embeddings)
        
        # sample diffusion
        df = sample_diff(table, run, is_cond=True, cond=cond, device=device, foreign_keys=foreign_keys, ids=ids[table])
        tables[table] = df
        print(f'Successfully sampled data for table {table}')
        print(df.head())

        if metadata.get_children(table):
            create_latent_embeddings(df, table, device = device)
    
    save_tables(tables, dataset_name, data_type=f'synthetic/ours/{run}')


############################################################################################


def main():
    # TODO: remove hardcoded arguments and run for both datasets
    dataset_name = 'rossmann-store-sales' 
    num_samples = 100
    cond = 'mlp'
    run = 'TEST'
    
    sample(dataset_name=dataset_name, num_samples=num_samples, run=run, cond=cond)


if __name__ == '__main__':
    main()
