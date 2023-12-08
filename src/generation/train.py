import os
import json

import numpy as np
import pandas as pd

from diffusion import train_diff
from autoencoder import train_vae
from utils_train import preprocess
from tabsyn.latent_utils import get_input_train
from src.data.utils import load_tables, load_metadata
from src.embedding_generation.generate_embeddings import get_rossmann_embeddings

def main():
    # args: HARDCODED for now TODO
    dataset_name = 'rossmann-store-sales'
    retrain_vae = False
    # read data
    metadata = load_metadata(dataset_name)
    tables = load_tables(dataset_name, 'train')
    # create graph
    # train GIN and compute structural embeddings using GIN
    print('Training GIN')
    get_rossmann_embeddings()
    # for each table in dataset
    for table in metadata.get_tables():
        # train vae
        data_dir = f'tabsyn/data/{table}'
        info_path = f'tabsyn/data/{table}/info.json'
        with open(info_path, 'r') as f:
            info = json.load(f)
        X_num, X_cat, categories, d_numerical = preprocess(data_dir, task_type = 'binclass', concat=False)
        if retrain_vae or not os.path.exists(f'ckpt/{table}/vae/decoder.pt'):
            print(f'Training VAE for table {table}')
            train_vae(X_num, X_cat, categories, d_numerical, info, epochs=4000)
        else:
            print(f'Reusing VAE for table {table}')
        # combine vae embeddings from parent tables with structural embeddings to condition diffusion
        gin_embeddings  =  pd.read_csv(f'data/gin_embeddings/rossmann-store-sales/{table}_embeddings.csv').to_numpy()
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
        
        np.save(f'ckpt/{table}/cond_train_z.npy', conditional_embeddings)

        # train conditional diffusion
        train_z, train_z_cond, _, ckpt_path, _ = get_input_train(table, is_cond=True)
        print(f'Training conditional diffusion for table {table}')
        train_diff(train_z, train_z_cond, ckpt_path, epochs=10000, is_cond=True, device='cuda:0')
    
# GENERATION
# sample skeletons from GraphRNN
# compute structural embeddings using GIN
# for each table in dataset
#     sample data from conditional diffusion

if __name__ == '__main__':
    main()