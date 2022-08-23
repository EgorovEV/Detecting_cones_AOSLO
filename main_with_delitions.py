import argparse
import pandas as pd
import wandb
from itertools import product
from tqdm import tqdm

from utils import *
from pipeline_with_delitions import find_blobs

import warnings
warnings.filterwarnings("ignore")
print("##### WARNING TURNED OFF #####")

def grid_parameters(parameters):
    for params in product(*parameters.values()):
        yield dict(zip(parameters.keys(), params))

def estimate_num_variant(parameters):
    variants = 0
    for params in product(*parameters.values()):
        variants += 1
    return variants


if __name__ == '__main__':
    GT_data = pd.read_csv('dataset/BAK1008L1_2020_07_02_11_56_18_AOSLO_788_V006_annotated_JLR_128_97_646_612.csv')
    GT_data = GT_data.to_numpy()
    img = cv2.imread('dataset/BAK1008L1_2020_07_02_11_56_18_AOSLO_788_V006_annotated_JLR_128_97_646_612.tiff', -1)
    img_colorful = cv2.imread('dataset/BAK1008L1_2020_07_02_11_56_18_AOSLO_788_V006_annotated_JLR_128_97_646_612.tiff')

    config_zoo = {
        'random_seed': [2022],
        'lambda_dist': [1.0],
        'lambda_blob': [1.0],
        'dist_alpha': [0.3],
        'dist_sigma': [0.3],
        'dist_n_neighbours': [2],
        'region': [
            {
            'x_min': 0,
            'x_max': 518,
            'y_min': 0,
            'y_max': 515
            }
        ],
        'boundary_size': [
            {
            'x': 15,
            'y': 15
            }
        ],
        'gradient_type': ['np_grad'],
        'epoch_num': [20],
        'n_particles_coeff': [1.],
        'metric_measure_freq': [1],
        'step_mode': ['discrete'], #discrete or contin
        'blobness_formula': ['custom'],# 'simple_div', 'custom'
        'write_gif': [False],
        'metric_algo': ['Chamfer'],
        'mu_dist_func': [5.84],
        'sigma_dist_func': [0.83],
        'lambda_dist_func': [-0.1],
        'dist_energy_type': ['pit'],
        'threshhold_dist_del': [2.1],
        'particle_call_window': [5],
        'particle_call_threshold': [0., 0.45],
        'init_blob_threshold': [5.],
        'reception_field_size': [15],
        'verbose_func': [False]
    }



    VERBOSE = False
    USE_WANDB = False # careful to use, not checked after changes!
    USE_WANDB_SWEEP = False

    # if USE_WANDB:
    #     wandb.init(project='cones_AOSLO', config=config_zoo)

    if USE_WANDB_SWEEP:
        config_default = {'random_seed': 2022, 'lambda_dist': 1., 'lambda_blob': 0.6,
                      'dist_alpha': 0.3, 'dist_sigma': 0.3, 'dist_n_neighbours': 2,
                      'region': {'x_min': 300, 'x_max': 328, 'y_min': 320, 'y_max': 345},
                      'boundary_size': {'x': 5, 'y': 5},
                      'gradient_type': 'np_grad',
                      'epoch_num': 100, 'n_particles_coeff': 1.0, 'metric_measure_freq': 1,
                      'step_mode': 'contin',  # discrete or contin
                      'blobness_formula': 'div_corrected',  # 'simple_div', 'custom', 'div_corrected'
                      'write_gif': False,
                      'metric_algo': 'Chamfer'
                      }
        wandb.init(project="cones_AOSLO_sweep", config=config_default)
        wandb_config = wandb.config

        find_blobs(wandb_config, GT_data, img, img_colorful, VERBOSE, USE_WANDB)
    else:


        print('Estimated num of variants:', estimate_num_variant(config_zoo))

        results = []

        for cur_config in tqdm(grid_parameters(config_zoo)):
            GT_data = pd.read_csv('dataset/BAK1008L1_2020_07_02_11_56_18_AOSLO_788_V006_annotated_JLR_128_97_646_612.csv')
            GT_data = GT_data.to_numpy()
            img = cv2.imread('dataset/BAK1008L1_2020_07_02_11_56_18_AOSLO_788_V006_annotated_JLR_128_97_646_612.tiff', -1)
            img_colorful = cv2.imread('dataset/BAK1008L1_2020_07_02_11_56_18_AOSLO_788_V006_annotated_JLR_128_97_646_612.tiff')
            try:
                results.append( find_blobs(cur_config, GT_data, img, img_colorful, VERBOSE, USE_WANDB) )
            except Exception as error_type:
                print('error occured', error_type)

        result_table = pd.DataFrame(results)
        result_table = result_table.sort_values('best_lsa_mean')
        result_table.to_csv("benchmarking_chamfer.csv", index=False)