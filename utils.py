import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance_matrix
from scipy import ndimage
import scipy
from sklearn.neighbors import NearestNeighbors
import time


def np_sigmoid(x):
    return 1 / (1 + np.exp(-x))


def GT_enumerate_from_zero(gt_data):
    gt_data -= 1.
    return gt_data


def bicycle(particle_position, max_val, min_val=0):
    particle_position = particle_position if particle_position < max_val else max_val - 1.
    particle_position = particle_position if particle_position >= min_val else min_val + 1
    return particle_position


def calc_distances(particles, GT_particles, n_neighbours):
    n_particles = GT_particles.shape[0]
    ### FIND NEAREST NEIGHBOURS, CALC DISTANCES ###
    x_nn = NearestNeighbors(n_neighbors=n_neighbours + 1, leaf_size=1, algorithm='kd_tree', metric='l2').fit(
        GT_particles)
    min_dists, neigh_ind = x_nn.kneighbors(particles)
    min_dists, neigh_ind = min_dists[:, 1:], neigh_ind[:, 1:]

    top_dist_vectors = np.zeros((n_particles, n_neighbours, 2))
    for particle_idx, particle_distances in enumerate(GT_particles):
        cur_neighbour_coord = particles[neigh_ind[particle_idx]]
        cur_particle_coord = np.tile(GT_particles[particle_idx], (n_neighbours, 1))
        top_dist_vectors[particle_idx] = cur_particle_coord - cur_neighbour_coord

    return neigh_ind, top_dist_vectors


def calc_dist_energy(particles, n_neighbours, alpha, sigma):
    def exp_vector_norm(x, y):
        return np.sqrt((x ** 2.) + (y ** 2.)) / (sigma ** 2.)

    top_dist_matr, top_dist_vectors = calc_distances(particles, particles, n_neighbours)

    exp_resulting_vectors = np.zeros((top_dist_vectors.shape[0], 2))
    for vector_idx, neigh_vectors in enumerate(top_dist_vectors):
        directions = []
        for vector in neigh_vectors:
            directions.append(vector / exp_vector_norm(*vector))
        directions = np.asarray(directions)
        exp_resulting_vectors[vector_idx] = alpha * np.sum(directions, axis=0)

    exp_resulting_vectors_modules = np.linalg.norm(exp_resulting_vectors, axis=1)
    return exp_resulting_vectors, exp_resulting_vectors_modules


def energy_pit_func3(x, y, mu, sigma, lambda_):
    dist = np.sqrt((x ** 2.) + (y ** 2.))
    if dist <= mu:
        return scipy.stats.expon(lambda_).pdf(dist) - scipy.stats.norm(mu, sigma).pdf(dist) + 0.48
    elif dist >= mu + 2 * sigma:
        return 0
    else:
        return scipy.stats.expon(lambda_).pdf(dist) + max(scipy.stats.norm(mu, sigma).pdf(dist) - 0.48, -0.1)


def energy_pit_func2(x, y, mu, sigma, lambda_):
    dist = np.sqrt((x ** 2.) + (y ** 2.))
    if dist <= mu:
        return scipy.stats.expon(lambda_).pdf(dist) - scipy.stats.norm(mu, sigma).pdf(dist) + 0.48
    else:
        return scipy.stats.expon(lambda_).pdf(dist) + scipy.stats.norm(mu, sigma).pdf(dist) - 0.48


def energy_pit_func1(x, y, mu, sigma, lambda_):
    dist = np.sqrt((x ** 2.) + (y ** 2.))
    return scipy.stats.expon(lambda_).pdf(dist) - scipy.stats.norm(mu, sigma).pdf(dist) + 0.48


def get_precomputed_dist_func(cur_config):
    reception_field_size = cur_config['reception_field_size']
    precomputed_dist_forces_field = np.zeros((reception_field_size, reception_field_size))
    for x in range(reception_field_size):
        for y in range(reception_field_size):
            dx_from_center = x - (reception_field_size // 2)  # -10,...,0,...,+10.
            dy_from_center = y - (reception_field_size // 2)
            precomputed_dist_forces_field[y][x] += energy_pit_func2(dx_from_center, dy_from_center,
                                                                    mu=cur_config['mu_dist_func'],
                                                                    sigma=cur_config[
                                                                        'sigma_dist_func'],
                                                                    lambda_=cur_config[
                                                                        'lambda_dist_func']
                                                                    )
    return precomputed_dist_forces_field


def calc_dist_energy_pit(particles, n_neighbours, mu, sigma, lambda_,
                         use_optimisation=False, is_stable_particle_index=None):
    t1 = time.time()
    top_dist_matr, top_dist_vectors = calc_distances(particles, particles, n_neighbours)
    print('dist_matr_calc=', time.time() - t1)
    t1 = time.time()

    exp_resulting_vectors = np.zeros((top_dist_vectors.shape[0], 2))
    for vector_idx, neigh_vectors in enumerate(top_dist_vectors):
        if use_optimisation:
            if is_stable_particle_index[vector_idx]:
                continue
        directions = []
        for vector in neigh_vectors:
            normed_vector = vector / np.linalg.norm(vector)
            directions.append(normed_vector * energy_pit_func2(vector[0], vector[1], mu, sigma, lambda_))
            # print(directions[-1])
        directions = np.asarray(directions)
        exp_resulting_vectors[vector_idx] = np.sum(directions, axis=0)
    print('energy_calc=', time.time() - t1)
    exp_resulting_vectors_modules = np.linalg.norm(exp_resulting_vectors, axis=1)
    return exp_resulting_vectors, exp_resulting_vectors_modules


def calc_dist_energy_pit_optim(particles, is_stable_particle_index, n_neighbours, mu, sigma, lambda_):
    return calc_dist_energy_pit(particles, n_neighbours, mu, sigma, lambda_,
                                use_optimisation=True, is_stable_particle_index=is_stable_particle_index)


def calc_blob_energy(R_b, particles):
    blob_energy = np.zeros(particles.shape[0])
    for idx, (x_pos, y_pos) in enumerate(particles):
        blob_energy[idx] = R_b[int(y_pos)][int(x_pos)]
    return blob_energy


def calc_grad(img):
    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    return laplacian


def calc_grad_field(img, grad_type):
    if grad_type == 'np_grad':
        g_on_x_axis, g_on_y_axis = np.gradient(img)
    elif grad_type == 'sobel':
        g_on_x_axis = ndimage.sobel(img, axis=0, mode='constant')
        g_on_y_axis = ndimage.sobel(img, axis=1, mode='constant')
    else:
        raise AttributeError('grad_type {0} is not implemented'.format(grad_type))

    g_on_y_axis = -g_on_y_axis
    grad_vector = np.stack((g_on_x_axis, g_on_y_axis), axis=2)
    return grad_vector


def visualize(img, particles, GT_data, is_save=False, img_name='test', save_dir='./examples/'):
    pass
    # plt.imshow(img)
    # plt.scatter(GT_data[:, 0], GT_data[:, 1], color='r', linewidths=0.4)
    # plt.scatter(particles[:, 0], particles[:, 1], color='#00FFFF', linewidths=0.3)
    # plt.show()


def visualize_all(img, GT_data, particles_to_gt, particles, gt_to_particles, metric, experiment_idx, dice, iteration,
                  save_dir='./examples/'):
    plt.figure(figsize=(30, 30))
    plt.imshow(img)
    metric_2_dist_map = {'1': 1.42, '2': 2.83}
    TP_idx = []
    FN_idx = []
    FP_idx = []
    for particles_to_gt_idx, dist in enumerate(particles_to_gt):
        if dist < metric_2_dist_map[metric]:
            TP_idx.append(particles_to_gt_idx)
        else:
            FN_idx.append(particles_to_gt_idx)

    for gt_to_particles_idx, dist in enumerate(gt_to_particles):
        if dist > metric_2_dist_map[metric]:
            FP_idx.append(gt_to_particles_idx)

    print('len(FP_idx)=', len(FP_idx))
    plt.scatter(GT_data[tuple(TP_idx), 0], GT_data[tuple(TP_idx), 1], color='g', linewidths=0.8)
    plt.scatter(GT_data[tuple(FN_idx), 0], GT_data[tuple(FN_idx), 1], color='r', linewidths=0.8)
    plt.scatter(particles[tuple(FP_idx), 0], particles[tuple(FP_idx), 1], color='#FFFFFF', linewidths=0.8)

    plt.title("dice={}, iter={}".format(dice, iteration))
    plt.savefig(save_dir + '/best_in_{0}_exp_{1}.png'.format(metric, experiment_idx),
                dpi=300)
    plt.clf()


def visualize_colorful(img_colorful, particles, is_save=False, img_name='test', save_dir='./examples/'):
    pass
    # image_particles = img_colorful.copy()
    # for particle in particles:
    #     # print((particle[0], particle[1]))
    #     cv2.circle(image_particles, (round(particle[0]), round(particle[1])), radius=0, color=(0, 255, 255),
    #                thickness=-1)
    # if is_save:
    #     cv2.imwrite(save_dir + img_name + '.jpg', image_particles)
    # plt.imshow(image_particles)
    # plt.show()


def visualize_wandb(img_colorful, particles, color='r'):
    pass
    # color_map = {'r': (255, 0, 0),
    #              'g': (0, 255, 0),
    #              'b': (0, 0, 255),
    #              }
    # image_particles = img_colorful.copy()
    # for particle in particles:
    #     cv2.circle(image_particles, (round(particle[0]), round(particle[1])), radius=1, color=color_map[color],
    #                thickness=-1)
    # return image_particles


def calc_dist_to_neares(p_from, p_to, n_neighbours=1, return_indexes=False):
    if n_neighbours == 1 and p_from.shape[0] == p_to.shape[0]:
        assert not (p_from[:, 0] == p_to[:, 0]).all()
        assert not (p_from[:, 1] == p_to[:, 1]).all()
    x_nn = NearestNeighbors(n_neighbors=n_neighbours, leaf_size=1, algorithm='kd_tree', metric='l2').fit(p_from)
    min_dists, indexes = x_nn.kneighbors(p_to)
    if return_indexes:
        return min_dists, indexes
    return min_dists


def calc_metrics(particles, GT_particles, gt_to_particles, particles_to_gt, mode='hangarian'):
    # top_dist_matr, _ = calc_distances(particles, GT_particles, n_neighbours=GT_particles.shape[0])
    # distances = np.sum(dist_matr, axis=1)
    if mode == 'hangarian':
        dist_matr = distance_matrix(GT_particles, particles)
        row_ind, col_ind = linear_sum_assignment(dist_matr)
        return dist_matr[row_ind, col_ind], \
               dist_matr[row_ind, col_ind].sum(), \
               np.mean(dist_matr[row_ind, col_ind]), \
               np.var(dist_matr[row_ind, col_ind])

    elif mode == 'Chamfer':
        return (gt_to_particles, particles_to_gt), \
               np.sum(gt_to_particles) + np.sum(particles_to_gt), \
               np.mean(gt_to_particles) + np.mean(particles_to_gt), \
               np.var(gt_to_particles) + np.var(particles_to_gt)
    else:
        raise ValueError


def del_close_particles(particles, is_stable_particle_index, R_b, cur_config, threshold_dist=2.):
    min_dists, indexes = calc_dist_to_neares(particles, particles, n_neighbours=2, return_indexes=True)
    dist_to_nearest_particle = min_dists[:, 1]
    to_delete = {idx: False for idx in range(particles.shape[0])}
    for dist, relation in zip(dist_to_nearest_particle, indexes):
        particle_idx_from = relation[0]
        particle_idx_to = relation[1]
        # if particle_from is "stable", don't delete it
        if is_stable_particle_index[particle_idx_from]:
            continue
        # if dist between particle is too small
        if dist < threshold_dist:
            # and none of that particles are already marked "deleted"
            if to_delete[particle_idx_from] == False and to_delete[particle_idx_to] == False:
                # take particle with less blobness energy (not gradient), and mark it "deleted"
                energy1 = R_b[int(particles[particle_idx_from][1])][int(particles[particle_idx_from][0])]
                energy2 = R_b[int(particles[particle_idx_to][1])][int(particles[particle_idx_to][0])]
                del_particle_idx = particle_idx_from if energy1 < energy2 else particle_idx_to
                to_delete[del_particle_idx] = True

    new_particles_amount = particles.shape[0] - sum(to_delete.values())
    new_particles = np.zeros((new_particles_amount, 2))
    new_is_stable_particle_index = [None for _ in range(new_particles_amount)]
    particle_idx = 0
    for idx, is_delete in to_delete.items():
        if not is_delete:
            new_is_stable_particle_index[particle_idx] = is_stable_particle_index[idx]
            new_particles[particle_idx] = particles[idx][:]
            particle_idx += 1

    if cur_config['verbose_func']:
        print("Delete #particles=", particles.shape[0] - new_particles.shape[0])

    return new_particles, to_delete, new_is_stable_particle_index


def is_particle_here(particles, x, y):
    for particle in particles:
        if int(particle[0]) == x and int(particle[1]) == y:
            return True
    return False


def calc_gravity_field(img_shape, particles, cur_config):
    gravity_field = np.zeros(img_shape)
    for x in range(img_shape[1]):
        for y in range(img_shape[0]):
            if is_particle_here(particles, x, y):
                gravity_field[y][x] = 1.
                continue
            particles_with_cur_place = np.vstack((particles, np.array([x, y])))

            exp_resulting_vectors, exp_resulting_vectors_modules = calc_dist_energy_pit(particles_with_cur_place,
                                                                                        n_neighbours=cur_config[
                                                                                            'dist_n_neighbours'],
                                                                                        mu=cur_config['mu_dist_func'],
                                                                                        sigma=cur_config[
                                                                                            'sigma_dist_func'],
                                                                                        lambda_=cur_config[
                                                                                            'lambda_dist_func']
                                                                                        )
            gravity_field[y][x] = exp_resulting_vectors_modules[-1]
    return gravity_field


def calc_gravity_field_optim(precomputed_dist_forces_field, img_shape, particles, cur_config):
    gravity_field = np.zeros(img_shape) - 0.48
    reception_field_size = cur_config['reception_field_size']
    half_rfs = reception_field_size // 2
    for particle in particles:
        center_x, center_y = int(particle[0]), int(particle[1])
        low_x, low_y = max(0, center_x - half_rfs), max(0, center_y - half_rfs)
        high_x, high_y = min(img_shape[1], center_x + half_rfs + 1), min(img_shape[0], center_y + half_rfs + 1)

        low_x_kernel, low_y_kernel = -1 * min(0, center_x - half_rfs), -1 * min(0, center_y - half_rfs)
        diff_x = max(0, (center_x + half_rfs + 1 - img_shape[1]))
        diff_y = max(0, (center_y + half_rfs + 1 - img_shape[0]))
        # print('===')
        # print(diff_x, diff_y)
        high_x_kernel, high_y_kernel = reception_field_size - diff_x, reception_field_size - diff_y

        # print(low_y_kernel,high_y_kernel,low_x_kernel,high_x_kernel)
        # print(low_y,high_y, low_x,high_x)
        # gravity_field[low_y:high_y, low_x:high_x] += precomputed_dist_forces_field
        gravity_field[low_y:high_y, low_x:high_x] = np.maximum(precomputed_dist_forces_field[low_y_kernel:high_y_kernel,
                                                               low_x_kernel:high_x_kernel],
                                                               gravity_field[low_y:high_y, low_x:high_x])

    # if is_particle_here(particles, x, y):
    #     gravity_field[y][x] = 1.
    #     continue

    return gravity_field


def adding_particles(particles, is_stable_particle_index, gravity_field, img_shape, cur_config):
    origin_img_borders = {'y': [cur_config['boundary_size']['y'], img_shape[0] - cur_config['boundary_size']['y']],
                          'x': [cur_config['boundary_size']['x'], img_shape[1] - cur_config['boundary_size']['x']]}
    step = cur_config['particle_call_window']
    particles_added = 0
    for pix_y in range(origin_img_borders['y'][0], origin_img_borders['y'][1], step):
        for pix_x in range(origin_img_borders['x'][0], origin_img_borders['x'][1], step):
            argmin_grav_in_window = np.unravel_index(gravity_field[pix_y:pix_y + step, pix_x:pix_x + step].argmin(),
                                                     gravity_field[pix_y:pix_y + step, pix_x:pix_x + step].shape)

            min_grav_in_window = gravity_field[pix_y + argmin_grav_in_window[0], pix_x + argmin_grav_in_window[1]]

            if abs(min_grav_in_window) < cur_config['particle_call_threshold']:
                particles = np.vstack(
                    (particles, np.array([pix_x + argmin_grav_in_window[1], pix_y + argmin_grav_in_window[0]])))
                particles_added += 1

    if cur_config['verbose_func']:
        print('#Particle Added:', particles_added)
    is_stable_particle_index += [0 for _ in range(particles.shape[0] - particles_added, particles.shape[0])]
    return particles, is_stable_particle_index


def initialize_high_prop_location(blobness_values, init_blob_threshold, img_shape, cur_config):
    # take only from cur image
    origin_img_borders = {'y': [cur_config['boundary_size']['y'], img_shape[0] - cur_config['boundary_size']['y']],
                          'x': [cur_config['boundary_size']['x'], img_shape[1] - cur_config['boundary_size']['x']]}

    origin_img_blob_val = blobness_values[origin_img_borders['y'][0]:origin_img_borders['y'][1],
                          origin_img_borders['x'][0]:origin_img_borders['x'][1]]
    # init in place with prob > current
    all_y, all_x = np.where(origin_img_blob_val > init_blob_threshold)
    all_y += cur_config['boundary_size']['y']
    all_x += cur_config['boundary_size']['x']

    particles = np.vstack((all_x, all_y)).T
    if cur_config['verbose_func']:
        print('Starting particles:', particles.shape)

    return particles


def calc_acc_metrics(particles, GT_particles, gt_to_particles, particles_to_gt, cur_config):
    dist_threshods = {'1': 1.42, '2': 2.83}

    particles_metrics = {'1': {'TP': 0, 'FP': 0, 'FN': 0, 'dice': None, 'precision': None, 'recall': None, },
                         '2': {'TP': 0, 'FP': 0, 'FN': 0, 'dice': None, 'precision': None, 'recall': None, }}

    # iterate over closest distances from particles to GT (shape: #GT)
    for dist in particles_to_gt:
        for dist_name, dist_threshod in dist_threshods.items():
            if dist < dist_threshod:
                particles_metrics[dist_name]['TP'] += 1
            else:
                # we did not predict that cone locates here => FN
                particles_metrics[dist_name]['FN'] += 1

    for dist in gt_to_particles:
        for dist_name, dist_threshod in dist_threshods.items():
            if dist > dist_threshod:
                # we predict that cone locates here, and it's false => FP
                particles_metrics[dist_name]['FP'] += 1

    for dist_name, dist_threshod in dist_threshods.items():
        TP = particles_metrics[dist_name]['TP']
        FP = particles_metrics[dist_name]['FP']
        FN = particles_metrics[dist_name]['FN']
        particles_metrics[dist_name]['dice'] = 2. * TP / (2. * TP + FP + FN)
        particles_metrics[dist_name]['recall'] = TP / (TP + FN + 0.)
        particles_metrics[dist_name]['precision'] = TP / (TP + FP + 0.)

    return particles_metrics
