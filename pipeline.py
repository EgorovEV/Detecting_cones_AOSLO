import matplotlib.pyplot as plt
import pandas as pd
import skimage
import wandb
from tqdm import tqdm

from utils import *
from image_transformation import crop_image, mirror_padding_image


def find_blobs(cur_config, GT_data, img, img_colorful, VERBOSE, USE_WANDB):

    result = {
        'best_lsa_mean': +np.inf,
        'cur_config': cur_config
    }

    np.random.seed(cur_config['random_seed'])

    GT_data = GT_enumerate_from_zero(GT_data)

    ### CROP REGIAON ###
    img, img_colorful, GT_data = crop_image(img, img_colorful, GT_data, cur_config['region'])

    ### CREATE MIRROR-BOUNDARY, TO FORCE PARTILES STAY ON ORIGINAL IMAGE ###
    img, GT_data = mirror_padding_image(img, cur_config['boundary_size'], GT_data)

    ### CREATE PARTICLE LOCATIONS ON IMG ###
    cur_config['n_particles'] = int(GT_data.shape[0] * cur_config['n_particles_coeff'])
    assert cur_config['n_particles'] >= GT_data.shape[0]
    particles = np.random.uniform(low=[cur_config['boundary_size']['x'], cur_config['boundary_size']['y']],
                                  high=[img.shape[1] - cur_config['boundary_size']['x'], img.shape[0] - cur_config['boundary_size']['y']],
                                  size=(cur_config['n_particles'], 2))

    ### GET PARTICLE BLOBNESS ENERGY ###
    hessian_m = skimage.feature.hessian_matrix(img, sigma=1, mode='constant', cval=0, order='rc')
    # eigs[i, j, k] contains the ith-largest eigenvalue at position (j, k).
    eigs = skimage.feature.hessian_matrix_eigvals(hessian_m)
    # pointwise division, multiplications
    # page 5(down) from "Multiscale Vessel Enhancement Filtering"
    # R_b - measure of "blobness"
    # R_b = np.divide(np.abs(eigs[0, :, :]), np.abs(eigs[1, :, :]))
    R_b = np.divide(np.abs(eigs[0, :, :]), np.abs(eigs[1, :, :]))

    ### need to think about outlier filtration -- they can send our far away
    threshold = 10.
    # print( np.sum([R_b > threshold]))
    # print( np.sum([R_b < 1.]))          #### why is lambda1 < lambda2 exists?!
    # assert np.sum([R_b < 0.99]) == 0        #### why is lambda1 < lambda2 exists?!

    R_b[R_b > threshold] = threshold
    R_b = R_b / threshold

    if VERBOSE:
        plt.imshow(R_b)
        plt.scatter(GT_data[:,0], GT_data[:,1], color='r', linewidths=0.3)
        plt.show()
    # print(R_b.shape)
    # print(R_b)
    # a=0/0
    # calc grad of "Blobness field"
    blobness_grad = calc_grad_field(R_b)
    # print(blobness_grad.shape)

    if VERBOSE:
        visualize(img, particles, GT_data, is_save=True,
                  img_name='start', save_dir='./examples/')

    CALC_STARTING_METRICS = False
    if CALC_STARTING_METRICS:
        exp_resulting_vectors, exp_resulting_vectors_modules = calc_dist_energy(particles,
                                                                                n_neighbours=cur_config['dist_n_neighbours'],
                                                                                alpha=cur_config['dist_alpha'],
                                                                                sigma=cur_config['dist_sigma']
                                                                                )
        blob_energy = calc_blob_energy(R_b, particles)

        cur_situation = visualize_wandb(img, particles, color='r')
        cur_and_GT_situation = visualize_wandb(img, GT_data, color='g')
        particles_diff_visual = wandb.Image(cur_and_GT_situation, caption="all_particles_location")

        logs = {'step': -1,
                'blob_energy': wandb.Histogram(blob_energy),
                'blob_energy_sum': np.sum(blob_energy),
                'blob_energy_mean': np.mean(blob_energy),
                'dist_energy_sum': np.sum(exp_resulting_vectors_modules),
                'dist_energy_mean': np.mean(exp_resulting_vectors_modules),
                'dist_energy': wandb.Histogram(exp_resulting_vectors_modules),
                'Particles location. Green-GT, Red-particles': particles_diff_visual
                }

        lsa, lsa_sum, lsa_mean, lsa_var = calc_metrics(particles, GT_data)
        logs['linear_sum_assignment'] = wandb.Histogram(lsa)
        logs['linear_sum_assignment_sum'] = lsa_sum
        logs['linear_sum_assignment_mean'] = lsa_mean
        logs['linear_sum_assignment_var'] = lsa_var
        wandb.log(logs)

    for iteration in range(cur_config['epoch_num']):
        exp_resulting_vectors, exp_resulting_vectors_modules = calc_dist_energy(particles,
                                                                                n_neighbours=cur_config['dist_n_neighbours'],
                                                                                alpha=cur_config['dist_alpha'],
                                                                                sigma=cur_config['dist_sigma']
                                                                                )
        for particle_idx, particle in enumerate(particles):
            # change x position of particle
            # print(blobness_grad.shape)
            # print(particle[1])
            # qwe = blobness_grad[int(particle[1])][int(particle[0])]
            particle[0] += -1 * cur_config['lambda_blob'] * blobness_grad[int(particle[1])][int(particle[0])][0] + \
                           cur_config['lambda_dist'] * exp_resulting_vectors[particle_idx][0]
            particle[0] = bicycle(particle[0], img.shape[1])  # to stay inside image borders

            # change y position of particle
            particle[1] += -1 * cur_config['lambda_blob'] * blobness_grad[int(particle[1])][int(particle[0])][1] + \
                           cur_config['lambda_dist'] * exp_resulting_vectors[particle_idx][1]
            particle[1] = bicycle(particle[1], img.shape[0])  # to stay inside image borders
            # qwe = blobness_grad[int(particle[1])][int(particle[0])]

        if USE_WANDB:
            blob_energy = calc_blob_energy(R_b, particles)

            cur_situation = visualize_wandb(img_colorful, particles, color='r')
            cur_and_GT_situation = visualize_wandb(cur_situation, GT_data, color='g')
            particles_diff_visual = wandb.Image(cur_and_GT_situation, caption="all_particles_location")

            logs = {'step': iteration,
                    'blob_energy': wandb.Histogram(blob_energy),
                    'blob_energy_sum': np.sum(blob_energy),
                    'blob_energy_mean': np.mean(blob_energy),
                    'dist_energy_sum': np.sum(exp_resulting_vectors_modules),
                    'dist_energy_mean': np.mean(exp_resulting_vectors_modules),
                    'dist_energy': wandb.Histogram(exp_resulting_vectors_modules),
                    'Particles location. Green-GT, Red-particles': particles_diff_visual
                    }

            if iteration % 5 == 0:
                lsa, lsa_sum, lsa_mean, lsa_var = calc_metrics(particles, GT_data)
                logs['linear_sum_assignment'] = wandb.Histogram(lsa)
                logs['linear_sum_assignment_sum'] = lsa_sum
                logs['linear_sum_assignment_mean'] = lsa_mean
                logs['linear_sum_assignment_var'] = lsa_var
            wandb.log(logs)

        else:
            # print('blob_energy= ', np.mean(calc_blob_energy(R_b, particles)))
            # print('dist_energy= ', np.sum(exp_resulting_vectors_modules))
            if iteration % cur_config['metric_measure_freq'] == 0:
                lsa, lsa_sum, lsa_mean, lsa_var = calc_metrics(particles, GT_data)
                if VERBOSE:
                    print('lsa_sum, lsa_mean, lsa_var: ', lsa_sum, lsa_mean, lsa_var)
                if result['best_lsa_mean'] > lsa_mean * 1.01:
                    result['best_lsa_mean'] = lsa_mean




        if iteration % cur_config['metric_measure_freq'] == 0 and VERBOSE:
            visualize(img, particles, GT_data, is_save=True,
                      img_name='step_' + str(iteration), save_dir='./examples/')

    if VERBOSE:
        visualize(img, particles, GT_data, is_save=True,
                  img_name='finish', save_dir='./examples/')

    return result


if __name__ == '__main__':
    cur_config = {'random_seed': 2022, 'lambda_dist': 0.05, 'lambda_blob': 0.01, 'dist_alpha': 0.3, 'dist_sigma': 0.3, 'dist_n_neighbours': 2, 'region': {'x_min': 300, 'x_max': 328, 'y_min': 320, 'y_max': 345}, 'boundary_size': {'x': 15, 'y': 15}, 'epoch_num': 500, 'n_particles_coeff': 2.0, 'metric_measure_freq': 100, 'n_particles': 52}


    GT_data = pd.read_csv('dataset/BAK1008L1_2020_07_02_11_56_18_AOSLO_788_V006_annotated_JLR_128_97_646_612.csv')
    GT_data = GT_data.to_numpy()
    img = cv2.imread('dataset/BAK1008L1_2020_07_02_11_56_18_AOSLO_788_V006_annotated_JLR_128_97_646_612.tiff', -1)
    img_colorful = cv2.imread('dataset/BAK1008L1_2020_07_02_11_56_18_AOSLO_788_V006_annotated_JLR_128_97_646_612.tiff')

    VERBOSE = True
    USE_WANDB = False

    find_blobs(cur_config, GT_data, img, img_colorful, VERBOSE, USE_WANDB)
