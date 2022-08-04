import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance_matrix
from scipy import ndimage
import scipy
from sklearn.neighbors import NearestNeighbors


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
    dist_matr = distance_matrix(GT_particles, particles)
    # return "n_neighbours" indexes. neigh_ind[i,j] - The index of j-closest element to element i
    neigh_ind = np.argsort(dist_matr, axis=1)[:, 1:n_neighbours + 1]

    top_dist_matr = np.zeros((n_particles, n_neighbours))
    top_dist_vectors = np.zeros((n_particles, n_neighbours, 2))
    for particle_idx, particle_distances in enumerate(dist_matr):
        for idx_of_closest_neighbour in range(n_neighbours):
            cur_neighbour_idx = neigh_ind[particle_idx][idx_of_closest_neighbour]
            top_dist_matr[particle_idx][idx_of_closest_neighbour] = dist_matr[particle_idx][cur_neighbour_idx]

            cur_neighbour_coord = particles[cur_neighbour_idx]
            cur_particle_coord = GT_particles[particle_idx]
            top_dist_vectors[particle_idx][idx_of_closest_neighbour] = cur_particle_coord - cur_neighbour_coord

    return top_dist_matr, top_dist_vectors


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


def calc_dist_energy_pit(particles, n_neighbours, mu, sigma, lambda_):
    def energy_pit_func2(x, y, mu, sigma, lambda_):
        dist = np.sqrt((x ** 2.) + (y ** 2.))
        if dist <= mu:
            return scipy.stats.expon(lambda_).pdf(dist) - scipy.stats.norm(mu, sigma).pdf(dist) + 0.48
        else:
            return scipy.stats.expon(lambda_).pdf(dist) + scipy.stats.norm(mu, sigma).pdf(dist) - 0.48

    def energy_pit_func(x, y, mu, sigma, lambda_):
        dist = np.sqrt((x ** 2.) + (y ** 2.))
        return scipy.stats.expon(lambda_).pdf(dist) - scipy.stats.norm(mu, sigma).pdf(dist) + 0.48

        # axis_diff = lambda dist: scipy.stats.expon(lambda_).pdf(dist) -\
        #         scipy.stats.norm(mu, sigma).pdf(dist)
        # dx, dy = np.sign(x) * axis_diff(abs(x)), np.sign(y) * axis_diff(abs(y))
        # return np.array([dx, dy])

    top_dist_matr, top_dist_vectors = calc_distances(particles, particles, n_neighbours)

    exp_resulting_vectors = np.zeros((top_dist_vectors.shape[0], 2))
    for vector_idx, neigh_vectors in enumerate(top_dist_vectors):
        directions = []
        for vector in neigh_vectors:
            normed_vector = vector / np.linalg.norm(vector)
            directions.append(normed_vector * energy_pit_func2(vector[0], vector[1], mu, sigma, lambda_))
            # print(directions[-1])
        directions = np.asarray(directions)
        exp_resulting_vectors[vector_idx] = np.sum(directions, axis=0)

    exp_resulting_vectors_modules = np.linalg.norm(exp_resulting_vectors, axis=1)
    return exp_resulting_vectors, exp_resulting_vectors_modules


def calc_blob_energy(R_b, particles):
    blob_energy = np.zeros(particles.shape[0])
    for idx, (x_pos, y_pos) in enumerate(particles):
        blob_energy[idx] = R_b[int(x_pos)][int(y_pos)]
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
    plt.imshow(img)
    plt.scatter(GT_data[:, 0], GT_data[:, 1], color='r', linewidths=0.4)
    plt.scatter(particles[:, 0], particles[:, 1], color='#00FFFF', linewidths=0.3)
    plt.show()


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


def calc_metrics(particles, GT_particles, mode='hangarian'):
    # top_dist_matr, _ = calc_distances(particles, GT_particles, n_neighbours=GT_particles.shape[0])
    dist_matr = distance_matrix(GT_particles, particles)
    # distances = np.sum(dist_matr, axis=1)
    if mode == 'hangarian':
        row_ind, col_ind = linear_sum_assignment(dist_matr)
        return dist_matr[row_ind, col_ind], \
               dist_matr[row_ind, col_ind].sum(), \
               np.mean(dist_matr[row_ind, col_ind]), \
               np.var(dist_matr[row_ind, col_ind])

    elif mode == 'Chamfer':
        gt_to_particles = calc_dist_to_neares(p_from=particles, p_to=GT_particles)
        particles_to_gt = calc_dist_to_neares(p_from=GT_particles, p_to=particles)
        return (gt_to_particles, particles_to_gt), \
               np.sum(gt_to_particles) + np.sum(particles_to_gt), \
               np.mean(gt_to_particles) + np.mean(particles_to_gt), \
               np.var(gt_to_particles) + np.var(particles_to_gt)
    else:
        raise ValueError


def update_particles(particles, R_b, threshold_dist=2.):
    min_dists, indexes = calc_dist_to_neares(particles, particles, n_neighbours=2, return_indexes=True)
    dist_to_nearest_particle = min_dists[:, 1]
    to_delete = {idx: False for idx in range(particles.shape[0])}
    for dist, relation in zip(dist_to_nearest_particle, indexes):
        particle_idx_from = relation[0]
        particle_idx_to = relation[1]
        # if dist between particle is too smal
        if dist < threshold_dist:
            # print('dist={}, idx_from={}, idx_to={}'.format(dist, particle_idx_from, particle_idx_to))
            # and none of that particles are already marked "deleted"
            if to_delete[particle_idx_from] == False and to_delete[particle_idx_to] == False:
                # take particle with less blobness energy (not gradient), and mark it "deleted"
                energy1 = R_b[int(particles[particle_idx_from][1])][int(particles[particle_idx_from][0])]
                energy2 = R_b[int(particles[particle_idx_to][1])][int(particles[particle_idx_to][0])]
                del_particle_idx = particle_idx_from if energy1 < energy2 else particle_idx_to
                to_delete[del_particle_idx] = True

    new_particles_amount = particles.shape[0] - sum(to_delete.values())
    new_particles = np.zeros((new_particles_amount, 2))
    particle_idx = 0
    for idx, is_delete in to_delete.items():
        if not is_delete:
            new_particles[particle_idx] = particles[idx][:]
            particle_idx += 1
    print("Old #particles={}, new #particles={}".format(particles.shape, new_particles.shape))
    # del particles
    return new_particles

def is_particle_here(particles, x, y):
    for particle in particles:
        if int(particle[0]) == x and int(particle[1]) == y:
            return True
    return False

def clac_grad_field(img_shape, particles, cur_config):
    gravity_field = np.zeros(img_shape)
    for x in range(img_shape[1]):
        for y in range(img_shape[0]):
            if is_particle_here(particles, x, y):
                gravity_field[y][x] = 1.
                continue
            particles_with_cur_place = np.vstack((particles, np.array([x, y])))
            # xx, yy = np.meshgrid(np.arange(gravity_field.shape[1]), np.arange(gravity_field.shape[0]))
            # # plt.quiver(x, y, -blobness_grad[:, :, 1], -blobness_grad[:, :, 0], )
            # plt.imshow(gravity_field, alpha=0.3)
            # plt.scatter(particles_with_cur_place[-1, 0], particles_with_cur_place[-1, 1], color='r', linewidths=0.8)
            # plt.scatter(particles_with_cur_place[:-2, 0], particles_with_cur_place[:-2, 1], color='#00FFFF', linewidths=0.8)
            # plt.show()

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


def adding_particles(particles, gravity_field, img_shape, cur_config):
    origin_img_borders = {'y': [cur_config['boundary_size']['y'], img_shape[0] - cur_config['boundary_size']['y']],
                          'x': [cur_config['boundary_size']['x'], img_shape[1] - cur_config['boundary_size']['x']]}
    step = cur_config['particle_call_window']
    particles_added = 0
    for pix_y in range(origin_img_borders['y'][0], origin_img_borders['y'][1], step):
        for pix_x in range(origin_img_borders['x'][0], origin_img_borders['x'][1], step):
            # argmin_grav_in_window = np.argmin(gravity_field[pix_y:pix_y+step, pix_x:pix_x+step]) # here can be an error if borders too tiny
            argmin_grav_in_window = np.unravel_index(gravity_field[pix_y:pix_y+step, pix_x:pix_x+step].argmin(),
                                                     gravity_field[pix_y:pix_y+step, pix_x:pix_x+step].shape)

            print('window: ', gravity_field[pix_y:pix_y+step, pix_x:pix_x+step],
                  'y1y2, x1x2:', pix_y,pix_y+step, pix_x,pix_x+step,
                  'argmin_grav_in_window', argmin_grav_in_window)

            min_grav_in_window = gravity_field[pix_y + argmin_grav_in_window[0], pix_x + argmin_grav_in_window[1]]
            print('min_grav=', min_grav_in_window)
            if min_grav_in_window < cur_config['particle_call_threshold']:
                particles = np.vstack((particles, np.array([pix_x + argmin_grav_in_window[1], pix_y + argmin_grav_in_window[0]])))
                particles_added += 1
    print('Particle Added:', particles_added)
    # print("particles shape:", particles.shape)
    xx, yy = np.meshgrid(np.arange(gravity_field.shape[1]), np.arange(gravity_field.shape[0]))
    # plt.quiver(x, y, -blobness_grad[:, :, 1], -blobness_grad[:, :, 0], )
    plt.imshow(gravity_field, alpha=0.3)
    plt.scatter(particles[-particles_added:, 0], particles[-particles_added:, 1], color='r', linewidths=0.8)
    plt.scatter(particles[:-particles_added-1, 0], particles[:-particles_added-1, 1], color='#00FFFF', linewidths=0.8)
    plt.show()


