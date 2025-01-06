import os
import pickle

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import zipfile
from .utils import *

def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def rolling_window(x, x_lag, add_batch_dim=True):
    if add_batch_dim:
        x = x[None, ...]
    return torch.cat([x[:, t:t + x_lag] for t in range(x.shape[1] - x_lag)], dim=0)


def get_data(data_type, p, q, **data_params):
    if data_type == 'STOCKS':
        scalers, x_real, x_real_scaled = get_equities_dataset(**data_params)
    else:
        raise NotImplementedError('Dataset %s not valid' % data_type)
    x_real_scaled = x_real_scaled.unsqueeze(0)
    assert x_real_scaled.shape[0] == 1
    x_real_scaled = rolling_window(x_real_scaled[0], p + q)
    return x_real_scaled, scalers


def get_equities_dataset(assets=None):
    # Step 1: Load and preprocess data
    df = pd.read_csv('./data/indices.csv')
    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
    df.set_index('Date', inplace=True)
    df = df.apply(pd.to_numeric).astype(float)
    log_returns = np.diff(np.log(df), axis=0)
    
    # Step 2: Compute log returns
    log_returns = np.diff(np.log(df), axis=0)
    print(log_returns.shape)

    # Step 3: Scale the log returns
    log_returns_scaled, scalers = scaling(log_returns)
    
    # Step 4: Prepare initial prices and create rolling windows
    log_returns_scaled = torch.from_numpy(log_returns_scaled).float()

    return scalers, log_returns, log_returns_scaled
