import numpy as np
import pandas as pd
import PIL
import cv2
import os
import matplotlib.pyplot as plt
import scipy
from scipy.stats import norm
from utils import calc_dist_to_neares


particles_amount = {}
resolution = []

particles_location_all_dataset = {}
for filename in os.listdir('all_dataset'):
    if '.csv' in filename:
        cur_data = pd.read_csv('all_dataset/'+filename)
        particles_amount[filename] = cur_data.shape[0]
        particles_location_all_dataset[filename]  = cur_data.to_numpy()
    if '.tiff' in filename:
        img = cv2.imread('dataset/BAK1008L1_2020_07_02_11_56_18_AOSLO_788_V006_annotated_JLR_128_97_646_612.tiff', -1)
        resolution.append(img.shape[:2])
# print(particles_amount)
# print(resolution)

plt.title('Distribution of the amount of particles')
plt.xlabel('Amount of particles')
plt.ylabel('Image amount')
plt.hist(particles_amount.values(), bins=41)
plt.show()

distances_betw_particles = {}
for image_name, one_image_particles in particles_location_all_dataset.items():
    distances_betw_particles[image_name] = calc_dist_to_neares(one_image_particles, one_image_particles, n_neighbours=2)[:, 1]

# for image_name, one_image_particles in particles_location_all_dataset.items():
#     plt.hist(distances_betw_particles[image_name],bins=10)
#     plt.title("Img={}, Amount of particles={}".format(image_name, particles_amount[image_name]))
#     plt.xlabel('Dist to neighbour')
#     plt.ylabel('Amount')
#     plt.show()


np_distances_betw_particles = np.zeros((1))
for dists in distances_betw_particles.values():
    np_distances_betw_particles = np.hstack((np_distances_betw_particles, np.array(dists)))
# mean_dist = np.mean(np_distances_betw_particles)
# var_dist = np.var(np_distances_betw_particles)

factor = 1/70000.
plt.hist(np_distances_betw_particles, bins=20,
         weights=factor*np.ones_like(np_distances_betw_particles),
         label='min dist between particles')
plt.xlim(2, 9)
plt.title("Min dist. Scaling_facor = {0:,.6f}".format(factor))

mu, sigma = norm.fit(np_distances_betw_particles)
lambda_ = 2
dist_energy_func = lambda x: +scipy.stats.expon(lambda_).pdf(x) -scipy.stats.norm(mu, sigma).pdf(x)
x = np.linspace(2,9, 100)
plt.plot(x, dist_energy_func(x),
         'r-', lw=5, alpha=0.6, label='energy function')
plt.legend()
plt.show()

