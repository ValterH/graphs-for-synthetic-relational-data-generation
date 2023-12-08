import os
import warnings
import time

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from tabsyn.model import MLPDiffusion, Model
from tabsyn.latent_utils import get_input_train, get_input_generate, recover_data, split_num_cat_target
from tabsyn.diffusion_utils import sample

warnings.filterwarnings('ignore')


def train_diff(train_z, train_z_cond, ckpt_path, epochs=4000, is_cond=False, device='cuda:0'): 

    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)

    in_dim = train_z.shape[1] 
    if is_cond:
        in_dim_cond = train_z_cond.shape[1]
    else:
        in_dim_cond = None

    mean, std = train_z.mean(0), train_z.std(0)

    train_z = (train_z - mean) / 2
    train_data = train_z

    if is_cond:
        train_data = torch.cat([train_z, train_z_cond], dim = 1)
    batch_size = 4096
    train_loader = DataLoader(
        train_data,
        batch_size = batch_size,
        shuffle = True,
        num_workers = 4,
    )


    denoise_fn = MLPDiffusion(in_dim, 1024, is_cond=is_cond, d_in_cond=in_dim_cond).to(device)
    print(denoise_fn)

    num_params = sum(p.numel() for p in denoise_fn.parameters())
    print("the number of parameters", num_params)

    model = Model(denoise_fn = denoise_fn, hid_dim = train_z.shape[1], is_cond=is_cond).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=20, verbose=True)

    model.train()

    best_loss = float('inf')
    patience = 0
    start_time = time.time()
    for epoch in range(epochs):
        
        pbar = tqdm(train_loader, total=len(train_loader))
        pbar.set_description(f"Epoch {epoch+1}/{epochs}")

        batch_loss = 0.0
        len_input = 0
        for batch in pbar:
            inputs = batch.float().to(device)
            loss = model(inputs)
        
            loss = loss.mean()

            batch_loss += loss.item() * len(inputs)
            len_input += len(inputs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix({"Loss": loss.item()})

        curr_loss = batch_loss/len_input
        scheduler.step(curr_loss)

        if curr_loss < best_loss:
            best_loss = loss.item()
            patience = 0
            torch.save(model.state_dict(), f'{ckpt_path}/model.pt')
        else:
            patience += 1
            if patience == 500:
                print('Early stopping')
                break

        if epoch % 1000 == 0:
            torch.save(model.state_dict(), f'{ckpt_path}/model_{epoch}.pt')

    end_time = time.time()
    print('Time: ', end_time - start_time)


def sample_diff(dataname, is_cond=True, device='cuda:0', num_samples=None, foreign_keys=None):

    if is_cond:
        cond_embedding_save_path = f'ckpt/{dataname}/gen/cond_z.npy'
        train_z_cond = torch.tensor(np.load(cond_embedding_save_path)).float()
        # TODO: this used to be train_z = train_z[:, 1:, :] <- the authors do not use the first token
        B, in_dim_cond = train_z_cond.size()
        train_z_cond = train_z_cond.view(B, in_dim_cond).to(device)
        num_samples = B
    else:
        train_z_cond = None
        in_dim_cond = None


    train_z, _, ckpt_path, info, num_inverse, cat_inverse = get_input_generate(dataname)
    in_dim = train_z.shape[1] 

    mean = train_z.mean(0)

    denoise_fn = MLPDiffusion(in_dim, 1024, is_cond=is_cond, d_in_cond=in_dim_cond).to(device)
    
    model = Model(denoise_fn = denoise_fn, hid_dim = train_z.shape[1], is_cond=is_cond).to(device)

    model.load_state_dict(torch.load(f'{ckpt_path}/model.pt'))

    '''
        Generating samples    
    '''
    start_time = time.time()

    sample_dim = in_dim

    x_next = sample(model.denoise_fn_D, num_samples, sample_dim, device=device, z_cond=train_z_cond)
    x_next = x_next * 2 + mean.to(device)

    syn_data = x_next.float().cpu().numpy()
    syn_num, syn_cat, syn_target = split_num_cat_target(syn_data, info, num_inverse, cat_inverse) 

    syn_df = recover_data(syn_num, syn_cat, syn_target, info)

    idx_name_mapping = info['idx_name_mapping']
    idx_name_mapping = {int(key): value for key, value in idx_name_mapping.items()}

    # convert data type
    for col in syn_df.columns:
        datatype = info['column_info'][str(col)]['subtype']
        if datatype == 'date':
            syn_df[col] = pd.to_datetime(syn_df[col].astype('int64')).dt.date
            continue
        syn_df[col] = syn_df[col].astype(datatype)
    syn_df.rename(columns = idx_name_mapping, inplace=True)
    
    # add fk column if conditional
    for fk, fk_values in foreign_keys.items():
        # add to the end of the dataframe
        syn_df.insert(len(syn_df.columns), fk, fk_values)
    # add id column
    syn_df.insert(info['id_col_idx'], info['id_col_name'], range(0, len(syn_df))) 
    
    end_time = time.time()
    print('Time:', end_time - start_time)
    
    return syn_df

if __name__ == '__main__':

    train_z, train_z_cond, _, ckpt_path, _ = get_input_train('store', is_cond=False)
    train_diff(train_z, train_z_cond, ckpt_path, epochs=10, is_cond=False, device='cuda:0')