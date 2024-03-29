import os
import time
import imageio
import pandas as pd
import skimage
import wandb

from image_transformation import crop_image, mirror_padding_image
from utils import *


def find_blobs(cur_config, GT_data, img, img_colorful, VERBOSE, USE_WANDB, experiment_idx=0):

    result = {
        'best_lsa_mean': +np.inf,
        'best_blobEn_mean': -np.inf,
        'dice_coeff_1': -np.inf,
        'dice_coeff_2': -np.inf,
        'cur_config': cur_config,
        'time_spended': 0,
    }
    t0 = time.time()

    np.random.seed(cur_config['random_seed'])

    GT_data = GT_enumerate_from_zero(GT_data)
    GT_data = np.round(GT_data)

    ### CROP REGIAON ###
    img, img_colorful, GT_data = crop_image(img, img_colorful, GT_data, cur_config['region'])

    ### CREATE MIRROR-BOUNDARY, TO FORCE PARTILES STAY ON ORIGINAL IMAGE ###
    img, GT_data = mirror_padding_image(img, cur_config['boundary_size'], GT_data)

    ### GET PARTICLE BLOBNESS ENERGY ###
    hessian_m = skimage.feature.hessian_matrix(img, sigma=1, mode='constant', cval=0, order='rc')
    # eigs[i, j, k] contains the ith-largest eigenvalue at position (j, k).
    eigs = skimage.feature.hessian_matrix_eigvals(hessian_m)

    ### PRECALCULATE POSSIBLE ONE-TO-ONE GRAVITY FORCES ###
    precomputed_dist_func_val = get_precomputed_dist_func(cur_config)

    assert (eigs[0, :, :] > eigs[1, :, :]).all()

    ####### EIGENVALUES VISUALISATION ###########
    if VERBOSE:
        fig, axis = plt.subplots()
        heatmap = axis.pcolor(eigs[1, :, :], cmap=plt.cm.YlGn)
        plt.scatter(GT_data[:, 0] + 0.5, GT_data[:, 1] + 0.5, color='r', linewidths=2.8)
        plt.colorbar(heatmap)
        plt.title('Eigenvalue 2')
        plt.show()

    if VERBOSE:
        plt.imshow(img)
        plt.scatter(GT_data[:, 0], GT_data[:, 1], color='r', linewidths=2.8)
        plt.title('GT in image')
        plt.show()

    if cur_config['blobness_formula'] == 'simple_div':
        R_b = np.divide(np.abs(eigs[0, :, :]), np.abs(eigs[1, :, :]))
        threshold = 10.
        print(np.sum([R_b > threshold]))
        R_b[R_b > threshold] = threshold
        R_b = R_b / threshold

    elif cur_config['blobness_formula'] == 'custom':
        R_b = eigs[1, :, :] - eigs[0, :, :] + (np_sigmoid(-eigs[0, :, :]) * 10.)  # + -eigs[0, :, :] * 100)

    elif cur_config['blobness_formula'] == 'div_corrected':
        idx_pos_l0 = [eigs[0, :, :] > 0.]
        idx_pos_l1 = [eigs[1, :, :] > 0.]
        idx_pos = idx_pos_l0 or idx_pos_l1
        max_abs_eigs = np.maximum(np.abs(eigs[0, :, :]), np.abs(eigs[1, :, :]))
        min_abs_eigs = np.minimum(np.abs(eigs[0, :, :]), np.abs(eigs[1, :, :]))
        R_b = np.divide(min_abs_eigs, max_abs_eigs)
        R_b[idx_pos] = 0.
        assert (R_b < 1.0001).all()
        assert (R_b > -0.0001).all()
    else:
        raise NotImplementedError

    if VERBOSE:
        fig, axis = plt.subplots()
        heatmap = axis.pcolor(R_b, cmap=plt.cm.YlGn)
        plt.colorbar(heatmap)
        plt.title('Blobness')
        plt.show()

    # calc grad of "Blobness field"
    blobness_grad = calc_grad_field(R_b, cur_config['gradient_type'])
    if VERBOSE:
        x, y = np.meshgrid(np.arange(blobness_grad.shape[1]), np.arange(blobness_grad.shape[0]))
        plt.quiver(x, y, -blobness_grad[:, :, 1], -blobness_grad[:, :, 0])
        plt.imshow(R_b, alpha=0.3)
        plt.scatter(GT_data[:, 0], GT_data[:, 1], color='r', linewidths=0.8)
        plt.title('Blobness gradient')
        plt.show()

    ### CREATE PARTICLE LOCATIONS ON IMG ###
    particles = initialize_high_prop_location(R_b,
                                              init_blob_threshold=cur_config['init_blob_threshold'],
                                              img_shape=img.shape,
                                              cur_config=cur_config)

    if VERBOSE:
        fig, axis = plt.subplots()
        heatmap = axis.pcolor(R_b, cmap=plt.cm.YlGn)
        plt.scatter(particles[:, 0] + 0.5, particles[:, 1] + 0.5, color='#000000', linewidths=0.8)
        plt.colorbar(heatmap)
        plt.title('Blobness with initialized stable particles')
        plt.show()

    is_stable_particle_index = [0 for _ in range(particles.shape[0])]

    while True:
        particles, idx_of_deleted_particles, is_stable_particle_index = \
            del_close_particles(particles, is_stable_particle_index, R_b, cur_config, threshold_dist=3)
        if sum(idx_of_deleted_particles.values()) == 0:
            break

    if VERBOSE:
        fig, axis = plt.subplots()
        heatmap = axis.pcolor(R_b, cmap=plt.cm.YlGn)
        plt.scatter(particles[:, 0] + 0.5, particles[:, 1] + 0.5, color='#000000', linewidths=0.8)
        plt.colorbar(heatmap)
        plt.title('Blobness with initialized stable particles')
        plt.show()

    is_stable_particle_index = [1 for _ in range(particles.shape[0])]

    if VERBOSE:
        fig, axis = plt.subplots()
        heatmap = axis.pcolor(R_b, cmap=plt.cm.YlGn)
        plt.scatter(particles[:, 0] + 0.5, particles[:, 1] + 0.5, color='#FFFFFF', linewidths=0.8)
        plt.colorbar(heatmap)
        plt.title('Blobness with initialized stable particles')
        plt.show()

    gravity_field = calc_gravity_field_optim(precomputed_dist_func_val, img.shape, particles, cur_config)

    if VERBOSE:
        fig, axis = plt.subplots()
        heatmap = axis.pcolor(gravity_field, cmap=plt.cm.YlGn)
        plt.scatter(particles[:, 0]+0.5, particles[:, 1]+0.5, color='#FFFFFF', linewidths=0.8)
        plt.colorbar(heatmap)
        plt.title('IPEnergy. Initialized stable particles')
        plt.show()

    particles, is_stable_particle_index = adding_particles(particles,
                                                           is_stable_particle_index,
                                                           gravity_field, img.shape, cur_config)

    if VERBOSE:
        fig, axis = plt.subplots()
        heatmap = axis.pcolor(gravity_field, cmap=plt.cm.YlGn)
        plt.scatter(particles[:, 0]+0.5, particles[:, 1]+0.5, color='#00FFFF', linewidths=0.8)
        is_stable_particle_index_bool = np.array(is_stable_particle_index).astype(bool)
        plt.scatter(particles[is_stable_particle_index_bool, 0]+0.5,
                    particles[is_stable_particle_index_bool, 1]+0.5, color='#FFFFFF', linewidths=0.8)
        plt.colorbar(heatmap)
        plt.title('IPEnergy. Stable particles - white, not_stable - cyan')
        plt.show()

    time_spended = {
        'dist_energy_calc': 0.,
        'moving_particles': 0.,
        'deleting_particles': 0.,
        'fix_resample': 0.,
        'metrics_logging': 0.
    }

    for iteration in range(cur_config['epoch_num']):
        t1 = time.time()
        print("Iter={}, #particles={}".format(iteration, particles.shape[0]))
        if cur_config['dist_energy_type'] == 'exp':
            exp_resulting_vectors, exp_resulting_vectors_modules = calc_dist_energy(particles,
                                                                                    n_neighbours=cur_config[
                                                                                        'dist_n_neighbours'],
                                                                                    alpha=cur_config['dist_alpha'],
                                                                                    sigma=cur_config['dist_sigma']
                                                                                    )

        elif cur_config['dist_energy_type'] == 'pit':
            exp_resulting_vectors, exp_resulting_vectors_modules = calc_dist_energy_pit_optim(particles,
                                                                                              is_stable_particle_index,
                                                                                              n_neighbours=cur_config[
                                                                                                  'dist_n_neighbours'],
                                                                                              mu=cur_config[
                                                                                                  'mu_dist_func'],
                                                                                              sigma=cur_config[
                                                                                                  'sigma_dist_func'],
                                                                                              lambda_=cur_config[
                                                                                                  'lambda_dist_func']
                                                                                              )
            if VERBOSE:
                fig, axis = plt.subplots()
                heatmap = axis.pcolor(gravity_field, cmap=plt.cm.YlGn)
                plt.scatter(particles[:, 0] + 0.5, particles[:, 1] + 0.5, color='#00FFFF', linewidths=0.8)
                is_stable_particle_index_bool = np.array(is_stable_particle_index).astype(bool)
                plt.scatter(particles[is_stable_particle_index_bool, 0] + 0.5,
                            particles[is_stable_particle_index_bool, 1] + 0.5, color='#FFFFFF', linewidths=0.8)
                for cur_particle_idx, (cur_particle, vector_force) in enumerate(zip(particles, exp_resulting_vectors)):
                    if not is_stable_particle_index_bool[cur_particle_idx]:
                        plt.arrow(x=cur_particle[0]+0.5, y=cur_particle[1]+0.5,
                                  dx=vector_force[0] * 3., dy=vector_force[1] * 3.,
                                  head_width=0.5)
                plt.colorbar(heatmap)
                plt.title('IPEnergy vectors. Stable particles - white, not_stable - cyan')
                plt.show()
        else:
            raise AttributeError('dist_energy_type={} is not implemented!'.format(cur_config['dist_energy_type']))
        time_spended['dist_energy_calc'] += time.time() - t1
        t1 = time.time()
        for particle_idx, particle in enumerate(particles):
            if is_stable_particle_index[particle_idx]:
                continue

            if cur_config['step_mode'] == 'contin':
                # Not supported scince july -- not optimal
                dy = -1 * cur_config['lambda_blob'] * blobness_grad[int(particle[1])][int(particle[0])][0] + \
                     -cur_config['lambda_dist'] * exp_resulting_vectors[particle_idx][0]
                dx = -1 * cur_config['lambda_blob'] * blobness_grad[int(particle[1])][int(particle[0])][1] + \
                     cur_config['lambda_dist'] * exp_resulting_vectors[particle_idx][1]
                dy = -dy
                particle[0] += dx
                particle[1] += dy

            elif cur_config['step_mode'] == 'discrete':  # and particle_idx == 0:
                delta_y = -1. * cur_config['lambda_blob'] * blobness_grad[int(particle[1])][int(particle[0])][0] + \
                          -cur_config['lambda_dist'] * exp_resulting_vectors[particle_idx][1]
                delta_x = -1. * cur_config['lambda_blob'] * blobness_grad[int(particle[1])][int(particle[0])][1] + \
                          cur_config['lambda_dist'] * exp_resulting_vectors[particle_idx][0]

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

        time_spended['moving_particles'] += time.time() - t1
        t1 = time.time()
        particles, _, is_stable_particle_index = \
            del_close_particles(particles, is_stable_particle_index, R_b, cur_config,
                                threshold_dist=cur_config['threshhold_dist_del'])
        assert particles.shape[0] == len(is_stable_particle_index)
        time_spended['deleting_particles'] += time.time() - t1
        t1 = time.time()

        if iteration % cur_config['fix_particles_frequency'] == 3:
            is_stable_particle_index = [1 for _ in range(len(is_stable_particle_index))]

            gravity_field = calc_gravity_field_optim(precomputed_dist_func_val, img.shape, particles, cur_config)

            particles, is_stable_particle_index = adding_particles(particles,
                                                                   is_stable_particle_index,
                                                                   gravity_field, img.shape,
                                                                   cur_config)
        time_spended['fix_resample'] += time.time() - t1
        t1 = time.time()
        if USE_WANDB:
            blob_energy = calc_blob_energy(R_b, particles)

            logs = {'step': iteration,
                    'blob_energy': wandb.Histogram(blob_energy),
                    'blob_energy_sum': np.sum(blob_energy),
                    'blob_energy_mean': np.mean(blob_energy)
                    }

            logs['GT_p_amount/particles_amount'] = len(GT_data) / len(particles)

            gt_to_particles = calc_dist_to_neares(GT_data, particles, n_neighbours=1, return_indexes=False)
            particles_to_gt = calc_dist_to_neares(particles, GT_data, n_neighbours=1, return_indexes=False)

            particles_metrics = calc_acc_metrics(particles, GT_data, gt_to_particles, particles_to_gt, cur_config, img.shape)
            for dist_threshold, particles_metrics_tr in particles_metrics.items():
                for metric_name, metric_val in particles_metrics_tr.items():
                    logs[dist_threshold + '_' + metric_name] = metric_val

            lsa, lsa_sum, lsa_mean, lsa_var = calc_metrics(particles, GT_data, gt_to_particles, particles_to_gt,
                                                           mode=cur_config['metric_algo'])
            # logs['linear_sum_assignment'] = wandb.Histogram(lsa)
            logs['linear_sum_assignment_sum'] = lsa_sum
            logs['linear_sum_assignment_mean'] = lsa_mean
            logs['linear_sum_assignment_var'] = lsa_var
            logs['amount_particles'] = len(particles)
            logs['amount_stable_particles'] = sum(is_stable_particle_index)
            logs['stable_particles_persentage'] = sum(is_stable_particle_index) / len(particles)
            wandb.log(logs)

            if result['dice_coeff_1'] < particles_metrics['1']['dice'] * 0.995:
                result['dice_coeff_1'] = particles_metrics['1']['dice']
                result['lsa_mean'] = lsa_mean
                result['lsa_var'] = lsa_var
                result['#GT/#particles'] = len(GT_data) / len(particles)
                result['experiment_idx'] = experiment_idx
                if cur_config['save_best_result_visualisation']:
                    visualize_all(img, GT_data, particles_to_gt, particles, gt_to_particles, metric='1',
                                  experiment_idx=experiment_idx, dice=result['dice_coeff_1'], iteration=iteration,
                                  cur_config=cur_config, img_shape=img.shape, save_dir='./examples/')

            if result['dice_coeff_2'] < particles_metrics['2']['dice'] * 0.995:
                result['dice_coeff_2'] = particles_metrics['2']['dice']
                result['lsa_mean'] = lsa_mean
                result['lsa_var'] = lsa_var
                result['#GT/#particles'] = len(GT_data) / len(particles)
                result['experiment_idx'] = experiment_idx
                if cur_config['save_best_result_visualisation']:
                    visualize_all(img, GT_data, particles_to_gt, particles, gt_to_particles, metric='2',
                                  experiment_idx=experiment_idx,
                                  dice=result['dice_coeff_2'], iteration=iteration, cur_config=cur_config, img_shape=img.shape, save_dir='./examples/')

        else:
            if iteration % cur_config['metric_measure_freq'] == 0:

                gt_to_particles = calc_dist_to_neares(GT_data, particles, n_neighbours=1, return_indexes=False)
                particles_to_gt = calc_dist_to_neares(particles, GT_data, n_neighbours=1, return_indexes=False)

                blob_energy = calc_blob_energy(R_b, particles)

                print('#GT/#particles =', len(GT_data) / len(particles))

                particles_metrics = calc_acc_metrics(particles, GT_data, gt_to_particles, particles_to_gt, cur_config, img.shape)
                for dist_threshold, particles_metrics_tr in particles_metrics.items():
                    for metric_name, metric_val in particles_metrics_tr.items():
                        print(dist_threshold + '_' + metric_name + ' = ' + str(metric_val), end=' | ')
                    print()

                if result['dice_coeff_1'] < particles_metrics['1']['dice'] * 0.995:
                    result['dice_coeff_1'] = particles_metrics['1']['dice']
                    if cur_config['save_best_result_visualisation']:
                        visualize_all(img, GT_data, particles_to_gt, particles, gt_to_particles, metric='1',
                                      experiment_idx=experiment_idx,
                                      dice=result['dice_coeff_1'], iteration=iteration, cur_config=cur_config, img_shape=img.shape, save_dir='./examples/')

                if result['dice_coeff_2'] < particles_metrics['2']['dice'] * 0.995:
                    result['dice_coeff_2'] = particles_metrics['2']['dice']
                    if cur_config['save_best_result_visualisation']:
                        visualize_all(img, GT_data, particles_to_gt, particles, gt_to_particles, metric='2',
                                      experiment_idx=experiment_idx,
                                      dice=result['dice_coeff_2'], iteration=iteration, cur_config=cur_config, img_shape=img.shape, save_dir='./examples/')

                lsa, lsa_sum, lsa_mean, lsa_var = calc_metrics(particles, GT_data, gt_to_particles, particles_to_gt,
                                                               mode=cur_config['metric_algo'])

                # if VERBOSE:
                print('lsa_sum, lsa_mean, lsa_var: ', lsa_sum, lsa_mean, lsa_var)
                if result['best_lsa_mean'] > lsa_mean:
                    result['best_lsa_mean'] = lsa_mean
                print("Stable particles percent= ", sum(is_stable_particle_index) / len(is_stable_particle_index))
                print('Blob energy sum, mean = ', np.sum(blob_energy), np.mean(blob_energy))
                if result['best_blobEn_mean'] < np.mean(blob_energy):
                    result['best_blobEn_mean'] = np.mean(blob_energy)
                x, y = np.meshgrid(np.arange(blobness_grad.shape[1]), np.arange(blobness_grad.shape[0]))
                plt.quiver(x, y, -blobness_grad[:, :, 1], -blobness_grad[:, :, 0], )
                plt.imshow(R_b, alpha=0.3)
                plt.scatter(GT_data[:, 0], GT_data[:, 1], color='r', linewidths=0.8)
                plt.scatter(particles[:, 0], particles[:, 1], color='#00FFFF', linewidths=0.8)
                is_stable_particle_index_bool = np.array(is_stable_particle_index).astype(bool)
                plt.scatter(particles[is_stable_particle_index_bool, 0],
                            particles[is_stable_particle_index_bool, 1], color='#FFFFFF', linewidths=0.8)
                plt.title('Particles over R_b. GT-red, stable - white, not_stable - cyan')
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

        if sum(is_stable_particle_index) / len(particles) > 0.997:
            break
        time_spended['metrics_logging'] += time.time() - t1

    if cur_config['write_gif']:

        with imageio.get_writer('mygif.gif', mode='I', fps=1) as writer:
            for filename in sorted(os.listdir('./img_log/')):
                image = imageio.v2.imread('img_log/' + filename)
                writer.append_data(image[:, :, :3])

    gt_to_particles = calc_dist_to_neares(GT_data, particles, n_neighbours=1, return_indexes=False)
    particles_to_gt = calc_dist_to_neares(particles, GT_data, n_neighbours=1, return_indexes=False)
    if cur_config['save_best_result_visualisation']:
        visualize_all(img, GT_data, particles_to_gt, particles, gt_to_particles, metric='1',
                      experiment_idx=experiment_idx,
                      dice=result['dice_coeff_2'], iteration=30, cur_config=cur_config, img_shape=img.shape, save_dir='./examples/')
        visualize_all(img, GT_data, particles_to_gt, particles, gt_to_particles, metric='2',
                      experiment_idx=experiment_idx,
                      dice=result['dice_coeff_2'], iteration=30, cur_config=cur_config, img_shape=img.shape, save_dir='./examples/')

    result['time_spended'] = time.time() - t0
    return result, time_spended


if __name__ == '__main__':

    VERBOSE = True
    USE_WANDB = False

    cur_config = {'random_seed': 2022, 'lambda_dist': 1., 'lambda_blob': 0.,
                  'dist_alpha': 0.3, 'dist_sigma': 0.3, 'dist_n_neighbours': 1,
                  # 'region': {'x_min': 300, 'x_max': 328, 'y_min': 320, 'y_max': 345},
                  # 'region': {'x_min': 310, 'x_max': 328, 'y_min': 330, 'y_max': 345},
                  # 'region': {'x_min': 0, 'x_max': 518, 'y_min': 0, 'y_max': 515},
                  'region': {'x_min': 50, 'x_max': 150, 'y_min': 50, 'y_max': 150},
                  'boundary_size': {'x': 10, 'y': 10},
                  'gradient_type': 'np_grad',
                  'epoch_num': 30, 'n_particles_coeff': 1.0, 'metric_measure_freq': 3,
                  'step_mode': 'discrete',  # discrete
                  'blobness_formula': 'custom',  # 'simple_div', 'custom', 'div_corrected'
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
                  'save_best_result_visualisation': True
                  }

    if cur_config['blobness_formula'] == 'div_corrected':
        cur_config['init_blob_threshold'] = 0.2
    if cur_config['blobness_formula'] == 'custom':
        cur_config['init_blob_threshold'] = 5.0

    GT_data = pd.read_csv('dataset/BAK1008L1_2020_07_02_11_56_18_AOSLO_788_V006_annotated_JLR_128_97_646_612.csv')
    GT_data = GT_data.to_numpy()
    img = cv2.imread('dataset/BAK1008L1_2020_07_02_11_56_18_AOSLO_788_V006_annotated_JLR_128_97_646_612.tiff', -1)
    img_colorful = cv2.imread('dataset/BAK1008L1_2020_07_02_11_56_18_AOSLO_788_V006_annotated_JLR_128_97_646_612.tiff')

    if USE_WANDB:
        wandb.init(project='cones_AOSLO', config=cur_config)

    _, time_spended = find_blobs(cur_config, GT_data, img, img_colorful, VERBOSE, USE_WANDB)

    print(time_spended)
