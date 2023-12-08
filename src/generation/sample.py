import json

import numpy as np
import pandas as pd

from diffusion import sample_diff
from autoencoder import create_latent_embeddings
from src.data.utils import load_tables, load_metadata
from src.data_modelling.pyg_datasets import sample_relational_distribution, get_rossmann_dataset
from src.embedding_generation.generate_embeddings import generate_embeddings


def main():
    # read data
    metadata = load_metadata('rossmann-store-sales')
    tables = dict()
    # create graph
    # TODO: actually sample a new graph not just load the original one
    dataset = get_rossmann_dataset()
    generate_embeddings(dataset, metadata, {'store': 0, 'test':1}, model_save_path = "models/gin_embeddings/rossmann-store-sales/", data_save_path="data/gin_embeddings/rossmann-store-sales/generated/")
    # compute structural embeddings using GIN
    pass
    # for each table in dataset
    for table in metadata.get_tables():
        
        gin_embeddings = pd.read_csv(f'data/gin_embeddings/rossmann-store-sales/generated/{table}_embeddings.csv').to_numpy()
        conditional_embeddings = [gin_embeddings]
        foreign_keys = dict()
        for parent in metadata.get_parents(table):
            parent_embeddings = []
            # parent embeddings are stored in the same order as the parent table
            parent_latent_embeddings = np.load(f'ckpt/{parent}/vae/gen_z.npy')
            parent_ids = tables[parent][metadata.get_primary_key(parent)].tolist()
            for fk in metadata.get_foreign_keys(parent, table):     
                # TODO get foregin keys from the graphs
                fks = pd.read_csv(f'data/gin_embeddings/rossmann-store-sales/generated/{table}_fks.csv').to_numpy()
                foreign_keys[fk] = fks
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

        np.save(f'ckpt/{table}/gen/cond_z.npy', conditional_embeddings)
        # sample diffusionge
        df = sample_diff(table, is_cond=True, device='cuda:0', foreign_keys=foreign_keys)
        tables[table] = df
        print(f'Successfully sampled data for table {table}')
        print(df.head())

        if metadata.get_children(table):
            create_latent_embeddings(df, table, device = 'cuda')

    
# GENERATION
# sample skeletons from GraphRNN
# compute structural embeddings using GIN
# for each table in dataset
#     sample data from conditional diffusion

if __name__ == '__main__':
    main()