import os
import warnings
from itertools import product

import pandas as pd
import traceback
import wandb
from tqdm import tqdm

from pipeline_with_delitions import find_blobs
from utils import *

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
        'lambda_dist': [0.1],
        'lambda_blob': [1.0],
        'dist_alpha': [0.3],
        'dist_sigma': [0.3],
        'dist_n_neighbours': [3],
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
                'x': 17,
                'y': 17
            }
        ],
        'gradient_type': ['np_grad'],
        'epoch_num': [30],
        'metric_measure_freq': [3],
        'step_mode': ['discrete'],  # discrete or contin
        'blobness_formula': ['custom'],  # 'simple_div', 'custom'
        'write_gif': [False],
        'metric_algo': ['Chamfer'],
        'mu_dist_func': [5.84],
        'sigma_dist_func': [0.83],
        'lambda_dist_func': [-0.1],
        'dist_energy_type': ['pit'],
        'threshhold_dist_del': [3.5],
        'particle_call_window': [5],
        'particle_call_threshold': [0.3],
        'init_blob_threshold': [5., ],
        'reception_field_size': [21],
        'fix_particles_frequency': [4],
        'verbose_func': [False],
        'save_best_result_visualisation': [False]
    }

    VERBOSE = False
    USE_WANDB = True
    USE_WANDB_SWEEP = True

    if USE_WANDB_SWEEP:
        default_config = {'random_seed': 2022, 'lambda_dist': 1., 'lambda_blob': 0.,
                          'dist_alpha': 0.3, 'dist_sigma': 0.3, 'dist_n_neighbours': 1,
                          # 'region': {'x_min': 300, 'x_max': 328, 'y_min': 320, 'y_max': 345},
                          # 'region': {'x_min': 310, 'x_max': 328, 'y_min': 330, 'y_max': 345},
                          # 'region': {'x_min': 300, 'x_max': 328, 'y_min': 320, 'y_max': 345},
                          'region': {'x_min': 0, 'x_max': 518, 'y_min': 0, 'y_max': 515},
                          'boundary_size': {'x': 10, 'y': 10},
                          'gradient_type': 'np_grad',
                          'epoch_num': 10, 'n_particles_coeff': 1.0, 'metric_measure_freq': 1,
                          'step_mode': 'discrete',  # discrete or contin
                          'blobness_formula': 'div_corrected',  # 'simple_div', 'custom', 'div_corrected'
                          'write_gif': False,
                          'metric_algo': 'Chamfer',
                          'mu_dist_func': 5.84,
                          'sigma_dist_func': 0.83,
                          'lambda_dist_func': -0.1,
                          'dist_energy_type': 'pit',
                          'threshhold_dist_del': 3.5,
                          'particle_call_window': 5,
                          'particle_call_threshold': 0.1,
                          'init_blob_threshold': 5.,
                          'reception_field_size': 21,
                          'fix_particles_frequency': 4,
                          'verbose_func': VERBOSE,
                          'save_best_result_visualisation': False
                          }

        print('Estimated num of variants:', estimate_num_variant(config_zoo))

        results = []
        experiment_idx = 0
        run = wandb.init(project='cones_AOSLO_all_dataset', config=default_config)
        cur_config = wandb.config

        print('~~~~~~~~~~~~~' * 5)
        print(cur_config)
        print('~~~~~~~~~~~~~' * 5)

        if cur_config['blobness_formula'] == 'div_corrected' and cur_config['init_blob_threshold'] > 1.:
            run.finish()
        if cur_config['blobness_formula'] == 'custom' and cur_config['init_blob_threshold'] < 1.:
            run.finish()

        try:
            result, _ = find_blobs(cur_config, GT_data, img, img_colorful, VERBOSE, USE_WANDB, experiment_idx)

            results.append(result)

        except Exception as error_type:
            print(traceback.format_exc())
            print('error occured', error_type)

    else:

        print('Estimated num of variants:', estimate_num_variant(config_zoo))

        results = []
        experiment_idx = 0
        for cur_config in tqdm(grid_parameters(config_zoo)):
            for filename in os.listdir('./all_dataset/'):
                if '.csv' in filename:
                    continue

                if cur_config['blobness_formula'] == 'div_corrected' and cur_config['init_blob_threshold'] > 1.:
                    continue
                if cur_config['blobness_formula'] == 'custom' and cur_config['init_blob_threshold'] < 1.:
                    continue

                if USE_WANDB:
                    run = wandb.init(project='cones_AOSLO_all_dataset', config=cur_config, reinit=True)

                GT_data = pd.read_csv('./all_dataset/' + filename[:-5] + '.csv')
                GT_data = GT_data.to_numpy()
                img = cv2.imread('./all_dataset/' + filename, -1)
                img_colorful = cv2.imread('./all_dataset/' + filename)

                print('./all_dataset/' + filename[:-5] + '.csv')
                print('./all_dataset/' + filename)

                try:
                    result, _ = find_blobs(cur_config, GT_data, img, img_colorful, VERBOSE, USE_WANDB, experiment_idx)

                    results.append(result)

                except Exception as error_type:
                    print(traceback.format_exc())
                    print('error occured', error_type)

                if USE_WANDB:
                    run.finish()
                experiment_idx += 1

        result_table = pd.DataFrame(results)
        result_table = result_table.sort_values('dice_coeff_1')
        result_table.to_csv("benchmarking_chamfer.csv", index=False)
