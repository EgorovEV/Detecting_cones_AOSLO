import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd
from scipy.spatial import distance_matrix


def bicycle(val, min_val=0, max_val=515):
    val = val if val < max_val else max_val - 1.
    val = val if val > min_val else min_val + 1
    return val


def metric():
    pass


def calc_distances(particles, n_neighbours):
    def get_particle_xy(idx):
        return particles[idx]

    n_particles = particles.shape[0]
    ### FIND NEAREST NEIGHBOURS, CALC DISTANCES ###
    dist_matr = distance_matrix(particles, particles)
    # return "n_neighbours" indexes. neigh_ind[i,j] - The index of j-closest element to element i
    neigh_ind = np.argsort(dist_matr, axis=1)[:, 1:n_neighbours + 1]

    top_dist_matr = np.zeros((n_particles, n_neighbours))
    top_dist_vectors = np.zeros((n_particles, n_neighbours, 2))
    for particle_idx, particle_distances in enumerate(dist_matr):
        for idx_of_closest_neighbour in range(n_neighbours):
            cur_neighbour_idx = neigh_ind[particle_idx][idx_of_closest_neighbour]
            top_dist_matr[particle_idx][idx_of_closest_neighbour] = dist_matr[particle_idx][cur_neighbour_idx]

            cur_neighbour_coord = particles[cur_neighbour_idx]
            cur_particle_coord = particles[particle_idx]
            top_dist_vectors[particle_idx][idx_of_closest_neighbour] = cur_particle_coord - cur_neighbour_coord

    return top_dist_matr, top_dist_vectors


def calc_dist_energy(particles, n_neighbours, alpha, sigma):
    def exp_vector_norm(x,y):
        return np.sqrt((x ** 2.) + (y ** 2.)) / (sigma ** 2.)

    top_dist_matr, top_dist_vectors = calc_distances(particles, n_neighbours)

    exp_resulting_vectors = np.zeros((top_dist_vectors.shape[0], 2))
    for vector_idx, neigh_vectors in enumerate(top_dist_vectors):
        directions = []
        for vector in neigh_vectors:
            directions.append( vector / exp_vector_norm(*vector) )
        directions = np.asarray(directions)
        exp_resulting_vectors[vector_idx] = alpha * np.sum(directions, axis=0)

    exp_resulting_vectors_modules = np.linalg.norm(exp_resulting_vectors)
    return exp_resulting_vectors, exp_resulting_vectors_modules


def calc_blob_energy(R_b, particles):
    blob_energy = np.zeros(particles.shape[0])
    for idx, (x_pos, y_pos) in enumerate(particles):
        blob_energy[idx] = R_b[int(x_pos)][int(y_pos)]
    return blob_energy


def calc_grad(img):
    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    return laplacian


def calc_grad_field(img):
    gx, gy = np.gradient(img)
    grad_vector = np.stack((gx, gy), axis=2)
    return grad_vector


def visualize(img_colorful, particles, is_save=False, img_name='test', save_dir='./examples/'):
    image_particles = img_colorful.copy()
    for particle in particles:
        # print((particle[0], particle[1]))
        cv2.circle(image_particles, (round(particle[0]), round(particle[1])), radius=1, color=(0, 255, 255),
                   thickness=-1)
    if is_save:
        cv2.imwrite(save_dir + img_name + '.jpg', image_particles)
    cv2.imshow('img with particles', image_particles)
    cv2.waitKey()
