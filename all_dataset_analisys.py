import numpy as np
import pandas as pd
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
print(particles_amount)
print(resolution)

plt.title('Distribution of the amount of particles')
plt.xlabel('Amount of particles')
plt.ylabel('Image amount')
plt.hist(particles_amount.values(), bins=41)
plt.show()

distances_betw_particles = {}
for image_name, one_image_particles in particles_location_all_dataset.items():
    distances_betw_particles[image_name] = calc_dist_to_neares(one_image_particles, one_image_particles, n_neighbours=2)[:, 1]



np_distances_betw_particles = np.zeros((1))
for dists in distances_betw_particles.values():
    np_distances_betw_particles = np.hstack((np_distances_betw_particles, np.array(dists)))

factor = 1/70000.
plt.hist(np_distances_betw_particles, bins=20,
         weights=factor*np.ones_like(np_distances_betw_particles),
         label='min dist between particles')
plt.xlim(0, 9)
plt.ylabel('Amount of pairs')
plt.xlabel('Distance (Pixel)')
plt.title("Y_axis Scaling_facor = {0:,.6f}".format(factor))

mu, sigma = norm.fit(np_distances_betw_particles)
print(mu, sigma)
lambda_ = -0.1
dist_energy_func1 = lambda x: +scipy.stats.expon(lambda_).pdf(x) -scipy.stats.norm(mu, sigma).pdf(x) +0.48

def dist_energy_func2(x, mu=5.841):
    ans = []
    for point in x:
        if point <= mu:
            ans.append(scipy.stats.expon(lambda_).pdf(point) -scipy.stats.norm(mu, sigma).pdf(point) +0.48)
        else:
            ans.append(scipy.stats.expon(lambda_).pdf(point) + scipy.stats.norm(mu, sigma).pdf(point) - 0.48)
    return ans

dist_energy_func0 = lambda x: 1/((x/(0.3**2))*0.3)

def dist_energy_func_paper(r, w=8.84, d=-0.1):
    ans = []
    for x in r:
        if x < w:
            ans.append( 1 + ((3*(d-1)*x)/w) - ((3*(d-1)*(x**2))/(w**2)) + ((d-1)*(x**3))/(w**3) )
        elif x > w and x < 11.:
            ans.append( d - ((2*d*((x-w)**3))/((w-1)**3)) - ((3*d*((x-w)**2))/((w-1)**2)) )
        else:
            ans.append( 0 )
    ans=np.array(ans)
    # ans[ans<-1.] = 0.
    return ans



x = np.linspace(-1.1,10, 100)
plt.plot(x, dist_energy_func2(x),
         'r-', lw=5, alpha=0.6, label='Inter-particle energy')
plt.legend()
plt.show()

