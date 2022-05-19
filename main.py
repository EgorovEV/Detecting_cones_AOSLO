import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import skimage
import wandb

from utils import *

VERBOSE = False
USE_WANDB = True

random_seed = 2022
lr = 1.
lambda_dist = 0.5 * lr
lambda_blob = 0.5 * lr
dist_alpha, dist_sigma = 1., 1.
dist_n_neighbours = 3

if USE_WANDB:
    wandb.init(project='cones_AOSLO', config={
        'lr': lr,
        'lambda_dist': lambda_dist,
        'lambda_blob': lambda_blob,
        'dist_alpha': dist_alpha,
        'dist_sigma': dist_sigma,
        'dist_n_neighbours': dist_n_neighbours,
        'random_seed': random_seed,
    })

np.random.seed(random_seed)

data = pd.read_csv('dataset/BAK1008L1_2020_07_02_11_56_18_AOSLO_788_V006_annotated_JLR_128_97_646_612.csv')
img = cv2.imread('dataset/BAK1008L1_2020_07_02_11_56_18_AOSLO_788_V006_annotated_JLR_128_97_646_612.tiff', -1)
img_colorful = cv2.imread('dataset/BAK1008L1_2020_07_02_11_56_18_AOSLO_788_V006_annotated_JLR_128_97_646_612.tiff')

### SHOW ORIGIN IMG ###
if VERBOSE:
    cv2.imshow('origin img', img)
    cv2.waitKey()

### CALC GRADINENT OVER IMG ###
grad = calc_grad(img)
if VERBOSE:
    plt.imshow(grad, cmap='gray')
    plt.show()

### CREATE PARTICLE LOCATIONS ON IMG ###
n_particles = 1000
particles = np.random.uniform(low=[0, 0], high=[img.shape[0], img.shape[1]], size=(n_particles, 2))

### SHOW PARTICLE LOCATIONS ON IMG ###
if VERBOSE:
    visualize(img_colorful, particles)

### GET PARTICLE DIST ENERGY ###
n_neighbours = 2
# particles_dist_energy = calc_dist_energy(particles, n_neighbours)

### GET PARTICLE BLOBNESS ENERGY ###
hessian_m = skimage.feature.hessian_matrix(img, sigma=1, mode='constant', cval=0, order='rc')
# eigs[i, j, k] contains the ith-largest eigenvalue at position (j, k).
eigs = skimage.feature.hessian_matrix_eigvals(hessian_m)
# pointwise division, multiplications
# page 5(down) from "Multiscale Vessel Enhancement Filtering"
# R_b - measure of "blobness"
R_b = np.divide(np.abs(eigs[0, :, :]), np.abs(eigs[1, :, :]))

# calc grad of "Blobness field"
blobness_grad = calc_grad_field(R_b)
print(blobness_grad.shape)

if VERBOSE:
    visualize(img_colorful, particles, is_save=True,
              img_name='start', save_dir='./examples/')

for iteration in range(200):

    exp_resulting_vectors, exp_resulting_vectors_modules = calc_dist_energy(particles,
                                                                            n_neighbours=dist_n_neighbours,
                                                                            alpha=dist_alpha,
                                                                            sigma=dist_sigma
                                                                            )

    for particle_idx, particle in enumerate(particles):
        particle[0] += -1 * lambda_blob * blobness_grad[int(particle[0])][int(particle[1])][0] + \
                       lambda_dist * exp_resulting_vectors[particle_idx][0]
        particle[0] = bicycle(particle[0])  # to stay inside image borders
        particle[1] += -1 * lambda_dist * blobness_grad[int(particle[0])][int(particle[1])][1] + \
                       lambda_dist * exp_resulting_vectors[particle_idx][1]
        particle[1] = bicycle(particle[1])



    if USE_WANDB:
        blob_energy = calc_blob_energy(R_b, particles)
        wandb.log({'step': iteration,
                   'blob_energy_sum': np.sum(blob_energy),
                   'blob_energy_mean': np.mean(blob_energy),
                   'dist_energy_sum': np.sum(exp_resulting_vectors_modules),
                   'dist_energy_mean': np.mean(exp_resulting_vectors_modules),
                   'dist_energy_hist': exp_resulting_vectors_modules})
        # wandb.log({'step': iteration, 'blob_energy_mean': np.mean(blob_energy)})
        # wandb.log({'step': iteration, 'dist_energy': calc_dist_energy(particles, n_neighbours=1)})
    else:

        print('blob_energy= ', np.mean(calc_blob_energy(R_b, particles)))
        print('dist_energy= ', np.sum(exp_resulting_vectors_modules))

    if iteration % 100 == 0 and VERBOSE:
        visualize(img_colorful, particles, is_save=True,
                  img_name='step_' + str(iteration), save_dir='./examples/')

if VERBOSE:
    visualize(img_colorful, particles, is_save=True,
              img_name='finish', save_dir='./examples/')

# if VERBOSE:
# plt.imshow(R_b, cmap = 'gray')
# plt.show()
