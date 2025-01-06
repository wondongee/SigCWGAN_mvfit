import itertools
import os
from os import path as pt

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import train_test_split

from hyperparameters import SIGCWGAN_CONFIGS
from lib import ALGOS
from lib.algos.base import BaseConfig
from lib.data import get_data
from lib.plot import savefig, create_summary
from lib.utils import *


def get_algo_config(dataset, data_params):
    """ Get the algorithms parameters. """
    key = dataset   
    if dataset == 'STOCKS':
        key += '_' + '_'.join(data_params['assets'])
    return SIGCWGAN_CONFIGS[key]

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

def get_algo(algo_id, base_config, dataset, data_params, x_real):
    if algo_id == 'SigCWGAN':
        algo_config = get_algo_config(dataset, data_params)
        algo = ALGOS[algo_id](x_real=x_real, config=algo_config, base_config=base_config)
    else:
        algo = ALGOS[algo_id](x_real=x_real, base_config=base_config)
    return algo


def run(algo_id, base_config, base_dir, dataset, spec, data_params={}):
    """ Create the experiment directory, calibrate algorithm, store relevant parameters. """
    print('Executing: %s, %s, %s' % (algo_id, dataset, spec))
    experiment_directory = pt.join(base_dir, dataset, spec, 'seed={}'.format(base_config.seed), algo_id)
    if not pt.exists(experiment_directory):        
        os.makedirs(experiment_directory)
        
    # Set seed for exact reproducibility of the experiments
    set_seed(base_config.seed)
    
    # initialise dataset and algo
    x_real, scalers = get_data(dataset, base_config.p, base_config.q, **data_params)
    x_real = x_real.to(base_config.device)
    
    ind_train = int(x_real.shape[0] * 0.7)
    x_real_train, x_real_test = x_real[:ind_train], x_real[ind_train:]

    algo = get_algo(algo_id, base_config, dataset, data_params, x_real_train)
    # Train the algorithm
    algo.fit()
    
    # Save the data
    x_fake = create_summary(dataset, base_config.device, algo.G, base_config.p, base_config.gen_len, x_real_test)
    pickle_path = os.path.join(experiment_directory, 'x_fake.pkl')  # 피클 파일 확장자 .pkl
    with open(pickle_path, 'wb') as f:
        pickle.dump(x_fake.cpu().numpy(), f)

    # Save the results    
    savefig('summary.png', experiment_directory)        
    # Pickle generator weights, real path and hyperparameters.
    pickle_it(x_real, pt.join(pt.dirname(experiment_directory), 'x_real.torch'))
    pickle_it(x_real_test, pt.join(pt.dirname(experiment_directory), 'x_real_test.torch'))
    pickle_it(x_real_train, pt.join(pt.dirname(experiment_directory), 'x_real_train.torch'))
    pickle_it(algo.training_loss, pt.join(experiment_directory, 'training_loss.pkl'))
    pickle_it(algo.G.to('cpu').state_dict(), pt.join(experiment_directory, 'G_weights.torch'))
    # Log some results at the end of training
    algo.plot_losses()
    savefig('losses.png', experiment_directory)


def get_dataset_configuration(dataset):        
    if dataset == 'STOCKS':
        generator = (('_'.join(asset), dict(assets=asset)) for asset in [('DJI', 'IXIC', 'JPM', 'HSI', 'GOLD', 'WTI')])
    else:
        raise Exception('%s not a valid data type.' % dataset)
    return generator

def main(args):                    
    args.use_cuda = True if torch.cuda.is_available() else False
    print('Start of training. CUDA: %s' % args.use_cuda)
    
    for dataset in args.datasets:
        for algo_id in args.algos:
            
            base_config = BaseConfig(
                device='cuda:{}'.format(args.device) if args.use_cuda and torch.cuda.is_available() else 'cpu',
                seed=args.seed,
                batch_size=args.batch_size,
                hidden_dims=args.hidden_dims,
                p=args.p,
                q=args.q,
                gen_len=args.gen_len,
                total_steps=args.total_steps,
                mc_samples=1000,
            )
            set_seed(args.seed)
            generator = get_dataset_configuration(dataset)
            for spec, data_params in generator:
                run(
                    algo_id=algo_id,
                    base_config=base_config,
                    data_params=data_params,
                    dataset=dataset,
                    base_dir=args.base_dir,
                    spec=spec,
                )


if __name__ == '__main__':
    import argparse
    print(os.getcwd())
    parser = argparse.ArgumentParser()
    # Meta parameters
    parser.add_argument('-base_dir', default='./numerical_results', type=str)
    parser.add_argument('-use_cuda', action='store_true')
    parser.add_argument('-device', default=0, type=int)    
    parser.add_argument('-seed', default=0, type=int)    
    parser.add_argument('-datasets', default=['STOCKS'], nargs="+") # default=['STOCK']
    parser.add_argument('-algos', default=['TimeGAN'], nargs="+") # default=['SigCWGAN', 'TimeGAN']
    

    # Algo hyperparameters
    parser.add_argument('-batch_size', default=200, type=int)
    parser.add_argument('-p', default=3, type=int)
    parser.add_argument('-q', default=3, type=int)
    parser.add_argument('-gen_len', default=128, type=int)
    parser.add_argument('-hidden_dims', default=3 * (50,), type=tuple)
    parser.add_argument('-total_steps', default=1000, type=int)

    args = parser.parse_args()        
    main(args)
