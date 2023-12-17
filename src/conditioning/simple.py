import numpy as np
import pandas as pd

def simple_message_passing(metadata, tables, table, embeddings_save_path, train=False):
    # combine vae embeddings from parent tables with structural embeddings from the current table to obtain condition diffusion
    gin_embeddings = pd.read_csv(f'{embeddings_save_path}{table}_embeddings.csv', index_col=0)
    ids = gin_embeddings.index.tolist()
    gin_embeddings = gin_embeddings.to_numpy()
    
    foreign_keys = dict()
    conditional_embeddings = [gin_embeddings]
    for parent in metadata.get_parents(table):
        parent_embeddings = []
        # parent embeddings are stored in the same order as the parent table
        if train:
            parent_latent_embeddings = np.load(f'ckpt/{parent}/vae/train_z.npy')
        else:
            parent_latent_embeddings = np.load(f'ckpt/{parent}/vae/gen_z.npy')
        parent_ids = tables[parent][metadata.get_primary_key(parent)].tolist()
        for fk in metadata.get_foreign_keys(parent, table):     
            if train:
                fks = tables[table][fk]
                foreign_keys[fk] = fks.tolist()
            else:
                fks = pd.read_csv(f'{embeddings_save_path}{table}_{fk}_fks.csv')
                foreign_keys[fk] = fks['parent'].tolist()
            cond_embeddings = np.zeros((len(fks), parent_latent_embeddings.shape[1], parent_latent_embeddings.shape[2]))
            for i, id in enumerate(foreign_keys[fk]):
                parent_idx = parent_ids.index(id)
                cond_embeddings[i] = parent_latent_embeddings[parent_idx]
            parent_embeddings.append(cond_embeddings)
        # average embeddings from different parents
        parent_embeddings = np.mean(parent_embeddings, axis=0)
        parent_embeddings = parent_embeddings.reshape((parent_embeddings.shape[0], -1))
        conditional_embeddings.append(parent_embeddings)
    if len(conditional_embeddings) > 1:
        conditional_embeddings = np.concatenate(conditional_embeddings, axis=1)
    else:
        conditional_embeddings = conditional_embeddings[0]
    
    original_ids = ids
    return conditional_embeddings, ids, original_ids, foreign_keys