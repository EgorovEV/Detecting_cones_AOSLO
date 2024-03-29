import os
import math
import matplotlib.pyplot as plt
import pandas as pd
import skimage
import wandb
import imageio
from PIL import Image
from tqdm import tqdm

from utils import *
from image_transformation import crop_image, mirror_padding_image


def find_blobs(cur_config, GT_data, img, img_colorful, VERBOSE, USE_WANDB):
    # if cur_config[''] > 10:


    result = {
        'best_lsa_mean': +np.inf,
        'cur_config': cur_config
    }

    np.random.seed(cur_config['random_seed'])

    GT_data = GT_enumerate_from_zero(GT_data)
    GT_data = np.round(GT_data)

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
    if cur_config['step_mode'] == 'discrete':
        particles = np.round(particles) #+ 0.5

    ### GET PARTICLE BLOBNESS ENERGY ###
    hessian_m = skimage.feature.hessian_matrix(img, sigma=1, mode='constant', cval=0, order='rc')
    # eigs[i, j, k] contains the ith-largest eigenvalue at position (j, k).
    eigs = skimage.feature.hessian_matrix_eigvals(hessian_m)


    assert (eigs[0, :, :] > eigs[1, :, :]).all()
    # eigs = np_sigmoid(eigs)

    # for row_idx, row in enumerate(eigs):
    #     for el_idx, el in enumerate(row):
    #         l1, l2 = el[0], el[1]
    #         if (l1 < 1 or l2 < 1) and :
    #             eigs[row_idx][el_idx][0]

    # assert (np.abs(eigs[0, :, :]) > np.abs((eigs[1, :, :]))).all()

    # first_eigs = np.maximum(eigs_mixes[0, :, :], eigs_mixes[1, :, :])
    # second_eigs = np.minimum(eigs_mixes[0, :, :], eigs_mixes[1, :, :])

    # pointwise division, multiplications
    # page 5(down) from "Multiscale Vessel Enhancement Filtering"
    # R_b - measure of "blobness"

    ######## EIGENVALUES VISUALISATION ###########
    # x, y = np.meshgrid(np.arange(eigs[0, :, :].shape[1]), np.arange(eigs[0, :, :].shape[0]))
    # # plt.quiver(x, y, eigs[0, :, :], eigs[0, :, :])
    # plt.imshow(eigs[0, :, :], alpha=0.3)
    # plt.scatter(GT_data[:,0], GT_data[:,1], color='r', linewidths=2.8)
    # plt.title('Lambda 1')

    ######## EIGENVALUES VISUALISATION ###########
    # fig, axis = plt.subplots()  # il me semble que c'est une bonne habitude de faire supbplots
    # heatmap = axis.pcolor(eigs[0, :, :], cmap=plt.cm.YlGn)
    # plt.scatter(GT_data[:, 0]+0.5, GT_data[:, 1]+0.5, color='r', linewidths=2.8)
    # plt.colorbar(heatmap)
    # plt.title('Lambda 1')
    # plt.show()

#
    # if VERBOSE:
    #     x, y = np.meshgrid(np.arange(eigs[1, :, :].shape[1]), np.arange(eigs[1, :, :].shape[0]))
    #     plt.quiver(x, y, eigs[1, :, :], eigs[1, :, :])
    #     plt.imshow(eigs[1, :, :], alpha=0.3)
    #     plt.scatter(GT_data[:,0], GT_data[:,1], color='r', linewidths=0.8)
    #     plt.show()

    if VERBOSE:
        plt.imshow(img)
        plt.scatter(GT_data[:, 0], GT_data[:, 1], color='r', linewidths=2.8)
        plt.show()

    if cur_config['blobness_formula'] == 'simple_div':
        R_b = np.divide(np.abs(eigs[0, :, :]), np.abs(eigs[1, :, :]))
        threshold = 10.
        print(np.sum([R_b > threshold]))
        ### need to think about outlier filtration -- they can send our far away
        R_b[R_b > threshold] = threshold
        R_b = R_b / threshold

    elif cur_config['blobness_formula'] == 'custom':
        R_b = eigs[1, :, :] - eigs[0, :, :] + (np_sigmoid(-eigs[0, :, :])*10.)# + -eigs[0, :, :] * 100)

    elif cur_config['blobness_formula'] == 'div_corrected':
        idx_pos_l0 = [eigs[0, :, :] > 0.]
        idx_pos_l1 = [eigs[1, :, :] > 0.]
        idx_pos = idx_pos_l0 or idx_pos_l1
        max_abs_eigs = np.maximum(np.abs(eigs[0, :, :]), np.abs(eigs[1, :, :]))
        min_abs_eigs = np.minimum(np.abs(eigs[0, :, :]), np.abs(eigs[1, :, :]))
        R_b = np.divide(min_abs_eigs, max_abs_eigs)
        R_b[idx_pos] = 0.
        assert (R_b<1.0001).all()
        assert (R_b>-0.0001).all()
    else:
        raise NotImplementedError

    # if VERBOSE:
    #     plt.imshow(R_b)
    #     plt.scatter(GT_data[:,0], GT_data[:,1], color='r', linewidths=0.3)
    #     plt.show()

    # calc grad of "Blobness field"
    blobness_grad = calc_grad_field(R_b, cur_config['gradient_type'])
    if VERBOSE:
        x, y = np.meshgrid(np.arange(blobness_grad.shape[1]), np.arange(blobness_grad.shape[0]))
        plt.quiver(x, y, -blobness_grad[:,:,1], -blobness_grad[:,:,0])
        plt.imshow(R_b, alpha=0.3)
        plt.scatter(GT_data[:,0], GT_data[:,1], color='r', linewidths=2.8)
        plt.show()

    if VERBOSE:
        visualize(img, particles, GT_data, is_save=True,
                  img_name='start', save_dir='./examples/')


    for iteration in range(cur_config['epoch_num']):
        if cur_config['dist_energy_type'] == 'exp':
            exp_resulting_vectors, exp_resulting_vectors_modules = calc_dist_energy(particles,
                                                                                    n_neighbours=cur_config['dist_n_neighbours'],
                                                                                    alpha=cur_config['dist_alpha'],
                                                                                    sigma=cur_config['dist_sigma']
                                                                                    )

        elif cur_config['dist_energy_type'] == 'pit':
            exp_resulting_vectors, exp_resulting_vectors_modules = calc_dist_energy_pit(particles,
                                                                                    n_neighbours=cur_config['dist_n_neighbours'],
                                                                                    mu=cur_config['mu_dist_func'],
                                                                                    sigma=cur_config['sigma_dist_func'],
                                                                                    lambda_=cur_config['lambda_dist_func']
                                                                                    )
            if VERBOSE:
                plt.imshow(R_b, alpha=0.3)
                for cur_particle, vector_force in zip(particles, exp_resulting_vectors):
                    plt.arrow(x=cur_particle[0], y=cur_particle[1],
                              dx=vector_force[0] * 3., dy=vector_force[1] * 3.,
                              head_width=0.5)
                plt.scatter(particles[:, 0], particles[:, 1], color='#00FFFF', linewidths=2.8)
                plt.show()
        else:
            raise AttributeError('dist_energy_type={} is not implemented!'.format(cur_config['dist_energy_type']))
        for particle_idx, particle in enumerate(particles):
            if cur_config['step_mode'] == 'contin':
                # change x position of particle
                dy = -1 * cur_config['lambda_blob'] * blobness_grad[int(particle[1])][int(particle[0])][0] + \
                               cur_config['lambda_dist'] * exp_resulting_vectors[particle_idx][0]
                dx = -1 * cur_config['lambda_blob'] * blobness_grad[int(particle[1])][int(particle[0])][1] + \
                     cur_config['lambda_dist'] * exp_resulting_vectors[particle_idx][1]

                #####
                # a,b = dx,dy
                # dx,dy=b,a
                dy = -dy
                #####
                particle[0] += dx
                particle[1] += dy

            elif cur_config['step_mode'] == 'discrete':# and particle_idx == 0:
                delta_y = -1. * cur_config['lambda_blob'] * blobness_grad[int(particle[1])][int(particle[0])][0] #+ \
                               # cur_config['lambda_dist'] * exp_resulting_vectors[particle_idx][0]
                delta_x = -1. * cur_config['lambda_blob'] * blobness_grad[int(particle[1])][int(particle[0])][1] #+ \
                               # cur_config['lambda_dist'] * exp_resulting_vectors[particle_idx][1]

                # delta_x, delta_y = delta_y, delta_x
                # a, b = delta_x, delta_y
                # delta_x, delta_y = b, a
                delta_y = -delta_y

                # print(delta_x, delta_y)
                if abs(delta_x) > abs(delta_y):
                    particle[0] += np.sign(delta_x)
                    if abs(delta_x) < abs(delta_y) * 2:
                        particle[1] += np.sign(delta_y)
                else:
                    particle[1] += np.sign(delta_y)
                    if abs(delta_y) < abs(delta_x) * 2:
                        particle[0] += np.sign(delta_x)

        if USE_WANDB:
            blob_energy = calc_blob_energy(R_b, particles)

            logs = {'step': iteration,
                    'blob_energy': wandb.Histogram(blob_energy),
                    'blob_energy_sum': np.sum(blob_energy),
                    'blob_energy_mean': np.mean(blob_energy),
                    'dist_energy_sum': np.sum(exp_resulting_vectors_modules),
                    'dist_energy_mean': np.mean(exp_resulting_vectors_modules),
                    }

            if iteration % 1 == 0:
                lsa, lsa_sum, lsa_mean, lsa_var = calc_metrics(particles, GT_data, mode=cur_config['metric_algo'])
                # logs['linear_sum_assignment'] = wandb.Histogram(lsa)
                logs['linear_sum_assignment_sum'] = lsa_sum
                logs['linear_sum_assignment_mean'] = lsa_mean
                logs['linear_sum_assignment_var'] = lsa_var
            wandb.log(logs)

        else:
            if iteration % cur_config['metric_measure_freq'] == 0:

                lsa, lsa_sum, lsa_mean, lsa_var = calc_metrics(particles, GT_data, mode=cur_config['metric_algo'])
                # if VERBOSE:
                print('lsa_sum, lsa_mean, lsa_var: ', lsa_sum, lsa_mean, lsa_var)
                if result['best_lsa_mean'] > lsa_mean * 1.01:
                    result['best_lsa_mean'] = lsa_mean


                x, y = np.meshgrid(np.arange(blobness_grad.shape[1]), np.arange(blobness_grad.shape[0]))
                plt.quiver(x, y, -blobness_grad[:, :, 1], -blobness_grad[:, :, 0], )
                plt.imshow(R_b, alpha=0.3)
                plt.scatter(GT_data[:, 0], GT_data[:, 1], color='r', linewidths=0.8)
                plt.scatter(particles[:, 0], particles[:, 1], color='#00FFFF', linewidths=0.8)
                if cur_config['write_gif']:
                        plt.savefig('img_log/{}_sum{}_mean{}_var{}.png'.format(iteration,
                                                                               lsa_sum,
                                                                               lsa_mean,
                                                                               lsa_var),
                                    dpi=100)
                        plt.clf()
                elif VERBOSE:
                    plt.show()
                else:
                    plt.clf()

    if cur_config['write_gif']:

        with imageio.get_writer('mygif.gif', mode='I', fps=1) as writer:
            for filename in sorted(os.listdir('./img_log/')):
                image = imageio.v2.imread('img_log/'+filename)
                writer.append_data(image[:,:,:3])
                # print(image.shape)




        # if iteration % cur_config['metric_measure_freq'] == 0 and VERBOSE:
        #     visualize(img, particles, GT_data, is_save=True,
        #               img_name='step_' + str(iteration), save_dir='./examples/')

    if VERBOSE:
        visualize(img, particles, GT_data, is_save=True,
                  img_name='finish', save_dir='./examples/')

    return result


if __name__ == '__main__':
    cur_config = {'random_seed': 2022, 'lambda_dist': 0.05, 'lambda_blob': 0.01, 'dist_alpha': 0.3, 'dist_sigma': 0.3, 'dist_n_neighbours': 2, 'region': {'x_min': 300, 'x_max': 328, 'y_min': 320, 'y_max': 345}, 'boundary_size': {'x': 15, 'y': 15}, 'gradient_type': 'np_grad', 'epoch_num': 500, 'n_particles_coeff': 2.0, 'metric_measure_freq': 100, 'n_particles': 52}
    # cur_config = {'random_seed': 2022, 'lambda_dist': 0.05, 'lambda_blob': 0.1, 'dist_alpha': 0.3, 'dist_sigma': 0.3, 'dist_n_neighbours': 2, 'region': {'x_min': 300, 'x_max': 328, 'y_min': 320, 'y_max': 345}, 'boundary_size': {'x': 5, 'y': 5}, 'gradient_type': 'np_grad', 'epoch_num': 500, 'n_particles_coeff': 2.0, 'metric_measure_freq': 100, 'n_particles': 52}
    # cur_config = {'random_seed': 2022, 'lambda_dist': 0.05, 'lambda_blob': 0.05, 'dist_alpha': 0.3, 'dist_sigma': 0.3, 'dist_n_neighbours': 2, 'region': {'x_min': 300, 'x_max': 328, 'y_min': 320, 'y_max': 345}, 'boundary_size': {'x': 15, 'y': 15}, 'gradient_type': 'np_grad', 'epoch_num': 500, 'n_particles_coeff': 2.0, 'metric_measure_freq': 100, 'n_particles': 52}

    # cur_config = {'random_seed': 2022, 'lambda_dist': 0.05, 'lambda_blob': 0.01, 'dist_alpha': 0.3, 'dist_sigma': 0.3,
    #               'dist_n_neighbours': 2, 'region': {'x_min': 0, 'x_max': 518, 'y_min': 0, 'y_max': 515},
    #               'boundary_size': {'x': 15, 'y': 15}, 'gradient_type': 'np_grad', 'epoch_num': 50,
    #               'n_particles_coeff': 2.0, 'metric_measure_freq': 10, 'n_particles': 52}
    cur_config = {'random_seed': 2022, 'lambda_dist': 1., 'lambda_blob': 0.6,
                  'dist_alpha': 0.3, 'dist_sigma': 0.3, 'dist_n_neighbours': 1,
                  # 'region': {'x_min': 300, 'x_max': 328, 'y_min': 320, 'y_max': 345},
                  'region': {'x_min': 0, 'x_max': 518, 'y_min': 0, 'y_max': 515},
                  'boundary_size': {'x': 10, 'y': 10},
                  'gradient_type': 'np_grad',
                  'epoch_num': 10, 'n_particles_coeff': 1.0, 'metric_measure_freq': 1,
                  'step_mode': 'discrete', #discrete or contin
                  'blobness_formula': 'custom',# 'simple_div', 'custom', 'div_corrected'
                  'write_gif': False,
                  'metric_algo': 'Chamfer',
                  'mu_dist_func': 5.84,
                  'sigma_dist_func': 0.83,
                  'lambda_dist_func': -0.1,
                  'dist_energy_type': 'pit'
                  }



    GT_data = pd.read_csv('dataset/BAK1008L1_2020_07_02_11_56_18_AOSLO_788_V006_annotated_JLR_128_97_646_612.csv')
    GT_data = GT_data.to_numpy()
    img = cv2.imread('dataset/BAK1008L1_2020_07_02_11_56_18_AOSLO_788_V006_annotated_JLR_128_97_646_612.tiff', -1)
    img_colorful = cv2.imread('dataset/BAK1008L1_2020_07_02_11_56_18_AOSLO_788_V006_annotated_JLR_128_97_646_612.tiff')

    VERBOSE = False
    USE_WANDB = False

    if USE_WANDB:
        wandb.init(project='cones_AOSLO', config=cur_config)

    find_blobs(cur_config, GT_data, img, img_colorful, VERBOSE, USE_WANDB)
