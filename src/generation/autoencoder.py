import os
import json
import time
import warnings

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau


from tqdm import tqdm

import tabsyn_utils as src
from vae.model import Model_VAE, Encoder_model, Decoder_model
from utils_train import preprocess, TabularDataset

warnings.filterwarnings('ignore')


LR = 1e-3
WD = 0
D_TOKEN = 4
TOKEN_BIAS = True

N_HEAD = 1
FACTOR = 32
NUM_LAYERS = 2


def compute_loss(X_num, X_cat, Recon_X_num, Recon_X_cat, mu_z, logvar_z):
    ce_loss_fn = nn.CrossEntropyLoss()
    mse_loss = (X_num - Recon_X_num).pow(2).mean()
    ce_loss = 0
    acc = 0
    total_num = 0

    for idx, x_cat in enumerate(Recon_X_cat):
        if x_cat is not None:
            ce_loss += ce_loss_fn(x_cat, X_cat[:, idx])
            x_hat = x_cat.argmax(dim = -1)
        acc += (x_hat == X_cat[:,idx]).float().sum()
        total_num += x_hat.shape[0]
    
    ce_loss /= (idx + 1)
    acc /= total_num
    # loss = mse_loss + ce_loss

    temp = 1 + logvar_z - mu_z.pow(2) - logvar_z.exp()

    loss_kld = -0.5 * torch.mean(temp.mean(-1).mean())
    return mse_loss, ce_loss, loss_kld, acc


def train_vae(X_num, X_cat, categories, d_numerical, info,  epochs=4000, lambd=0.7, max_beta = 0.01, min_beta=1e-5,  device='cuda:0'):
    ckpt_dir = f'ckpt/{info["name"]}/vae' 

    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    model_save_path = f'{ckpt_dir}/model.pt'
    encoder_save_path = f'{ckpt_dir}/encoder.pt'
    decoder_save_path = f'{ckpt_dir}/decoder.pt'

    X_train_num, _ = X_num
    X_train_cat, _ = X_cat

    X_train_num, X_test_num = X_num
    X_train_cat, X_test_cat = X_cat

    X_train_num, X_test_num = torch.tensor(X_train_num).float(), torch.tensor(X_test_num).float()
    X_train_cat, X_test_cat =  torch.tensor(X_train_cat), torch.tensor(X_test_cat)


    train_data = TabularDataset(X_train_num.float(), X_train_cat)

    X_test_num = X_test_num.float().to(device)
    X_test_cat = X_test_cat.to(device)

    batch_size = 4096
    train_loader = DataLoader(
        train_data,
        batch_size = batch_size,
        shuffle = True,
        num_workers = 4,
    )

    model = Model_VAE(NUM_LAYERS, d_numerical, categories, D_TOKEN, n_head = N_HEAD, factor = FACTOR, bias = True)
    model = model.to(device)

    pre_encoder = Encoder_model(NUM_LAYERS, d_numerical, categories, D_TOKEN, n_head = N_HEAD, factor = FACTOR).to(device)
    pre_decoder = Decoder_model(NUM_LAYERS, d_numerical, categories, D_TOKEN, n_head = N_HEAD, factor = FACTOR).to(device)

    pre_encoder.eval()
    pre_decoder.eval()

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.95, patience=10, verbose=True)

    best_train_loss = float('inf')

    current_lr = optimizer.param_groups[0]['lr']
    patience = 0

    beta = max_beta
    start_time = time.time()
    for epoch in range(epochs):
        pbar = tqdm(train_loader, total=len(train_loader))
        pbar.set_description(f"Epoch {epoch+1}/{epochs}")

        curr_loss_multi = 0.0
        curr_loss_gauss = 0.0
        curr_loss_kl = 0.0

        curr_count = 0

        for batch_num, batch_cat in pbar:
            model.train()
            optimizer.zero_grad()

            batch_num = batch_num.to(device)
            batch_cat = batch_cat.to(device)

            Recon_X_num, Recon_X_cat, mu_z, std_z = model(batch_num, batch_cat)
        
            loss_mse, loss_ce, loss_kld, train_acc = compute_loss(batch_num, batch_cat, Recon_X_num, Recon_X_cat, mu_z, std_z)

            loss = loss_mse + loss_ce + beta * loss_kld
            loss.backward()
            optimizer.step()

            batch_length = batch_num.shape[0]
            curr_count += batch_length
            curr_loss_multi += loss_ce.item() * batch_length
            curr_loss_gauss += loss_mse.item() * batch_length
            curr_loss_kl    += loss_kld.item() * batch_length

        num_loss = curr_loss_gauss / curr_count
        cat_loss = curr_loss_multi / curr_count
        kl_loss = curr_loss_kl / curr_count
        
        train_loss = num_loss + cat_loss
        scheduler.step(train_loss)

        new_lr = optimizer.param_groups[0]['lr']

        if new_lr != current_lr:
            current_lr = new_lr
            print(f"Learning rate updated: {current_lr}")
            
        if train_loss < best_train_loss:
            best_train_loss = train_loss
            patience = 0
            torch.save(model.state_dict(), model_save_path)
        else:
            patience += 1
            if patience == 10:
                if beta > min_beta:
                    beta = beta * lambd

        '''
            Evaluation
        '''
        model.eval()
        with torch.no_grad():
            Recon_X_num, Recon_X_cat, mu_z, std_z = model(X_test_num, X_test_cat)

            val_mse_loss, val_ce_loss, val_kl_loss, val_acc = compute_loss(X_test_num, X_test_cat, Recon_X_num, Recon_X_cat, mu_z, std_z)
            val_loss = val_mse_loss.item() * 0 + val_ce_loss.item()

            scheduler.step(val_loss)

        print('epoch: {}, beta = {:.6f}, Train MSE: {:.6f}, Train CE:{:.6f}, Train KL:{:.6f}, Val MSE:{:.6f}, Val CE:{:.6f}, Train ACC:{:6f}, Val ACC:{:6f}'.format(epoch, beta, num_loss, cat_loss, kl_loss, val_mse_loss.item(), val_ce_loss.item(), train_acc.item(), val_acc.item() ))

    end_time = time.time()
    print('Training time: {:.4f} mins'.format((end_time - start_time)/60))
    
    # Saving latent embeddings
    with torch.no_grad():
        pre_encoder.load_weights(model)
        pre_decoder.load_weights(model)

        torch.save(pre_encoder.state_dict(), encoder_save_path)
        torch.save(pre_decoder.state_dict(), decoder_save_path)

        X_train_num = X_train_num.to(device)
        X_train_cat = X_train_cat.to(device)

        X_test_cat = X_test_cat.to(device)
        X_test_num = X_test_num.to(device)

        print('Successfully load and save the model!')

        train_z = pre_encoder(X_train_num, X_train_cat).detach().cpu().numpy()

        test_z = pre_encoder(X_test_num, X_test_cat).detach().cpu().numpy()

        np.save(f'{ckpt_dir}/train_z.npy', train_z)
        np.save(f'{ckpt_dir}/test_z.npy', test_z)

        print('Successfully save pretrained embeddings in disk!')



def create_latent_embeddings(df, dataname, device = 'cuda'):
    info_path = f'tabsyn/data/{dataname}/info.json'
    data_dir = f'tabsyn/data/{dataname}'

    with open(info_path, 'r') as f:
        info = json.load(f)

    column_names = df.columns
    num_col_idx = info['num_col_idx']
    cat_col_idx = info['cat_col_idx']
    num_columns = [column_names[i] for i in num_col_idx]
    cat_columns = [column_names[i] for i in cat_col_idx]

    # replace nan categorical values with a string 'nan
    df_cat = df[cat_columns]

    df_cat.fillna('nan', inplace=True)

    # ensure the data is encoded in the sam way as the training data
    X_num_t, X_cat_t, y_t = src.read_pure_data(data_dir, 'train')

    X_num = df[num_columns].to_numpy().astype(np.float32)
    X_cat = df_cat.to_numpy()

    # original data is in the train split
    # the data to encode is in the test split
    assert X_cat_t.shape == X_cat.shape
    assert X_num_t.shape == X_num.shape
    
    X_categorical = {'train': X_cat_t, 'test': X_cat}
    X_numerical = {'train': X_num_t, 'test': X_num}
    y = {'train': y_t,     'test': np.zeros(X_num.shape[0])}

    T_dict = {}
    T_dict['normalization'] = "quantile"
    T_dict['num_nan_policy'] = 'mean'
    T_dict['cat_nan_policy'] =  None
    T_dict['cat_min_frequency'] = None
    T_dict['cat_encoding'] = None
    T_dict['y_policy'] = "default"

    T = src.Transformations(**T_dict)

    D = src.Dataset(
        X_numerical,
        X_categorical,
        y,
        y_info={},
        task_type=src.TaskType(info['task_type']),
        n_classes=info.get('n_classes')
    )
    dataset = src.transform_dataset(D, T, None)

    # our dataset is contained in the test split
    X_num, X_cat = dataset.X_num['test'], dataset.X_cat['test']

    categories = src.get_categories(dataset.X_cat['train'])
    d_numerical = X_num.shape[1]

    X_num = torch.tensor(X_num).float()
    X_cat =  torch.tensor(X_cat)
    
    X_num = X_num.to(device)
    X_cat = X_cat.to(device)

    # Load model
    pre_encoder = Encoder_model(NUM_LAYERS, d_numerical, categories, D_TOKEN, n_head = N_HEAD, factor = FACTOR).to(device)
    pre_encoder.load_state_dict(torch.load(f'ckpt/{dataname}/vae/encoder.pt'))

    with torch.no_grad():
        latents = pre_encoder(X_num, X_cat).detach().cpu().numpy()

    # save the latent embeddings
    np.save(f'ckpt/{dataname}/vae/gen_z.npy', latents)


if __name__ == '__main__':
    data_dir = 'tabsyn/data/store'
    info_path = f'tabsyn/data/store/info.json'
    with open(info_path, 'r') as f:
        info = json.load(f)
    X_num, X_cat, categories, d_numerical = preprocess(data_dir, task_type = 'binclass', concat=False)
    train_vae(X_num, X_cat, categories, d_numerical, info, epochs=3)
    