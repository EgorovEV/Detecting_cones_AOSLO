import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance_matrix
from scipy import ndimage
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

def metric():
    pass


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
        g_on_x_axis = ndimage.sobel(img,axis=0,mode='constant')
        g_on_y_axis = ndimage.sobel(img,axis=1,mode='constant')
    else:
        raise AttributeError('grad_type {0} is not implemented'.format(grad_type))

    g_on_y_axis = -g_on_y_axis
    grad_vector = np.stack((g_on_x_axis, g_on_y_axis), axis=2)
    return grad_vector

def visualize(img, particles, GT_data, is_save=False, img_name='test', save_dir='./examples/'):
    plt.imshow(img)
    plt.scatter(GT_data[:, 0], GT_data[:, 1], color='r', linewidths=0.4)
    plt.scatter(particles[:,0], particles[:,1], color='#00FFFF', linewidths=0.3)
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

def calc_dist_to_neares(p_from, p_to):
    x_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric='l2').fit(p_from)
    min_dists = x_nn.kneighbors(p_to)[0]
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

