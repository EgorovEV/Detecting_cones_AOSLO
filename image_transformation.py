import cv2
import pandas as pd
import numpy as np


def crop_image(img, img_colorful, gt_particles, region):
    assert (0 <= region['x_min']) and (0 <= region['y_min'])
    assert (img.shape[1] > region['x_max']) and (img.shape[0] > region['y_max'])

    img_crop = img[region['y_min']:region['y_max'], region['x_min']:region['x_max']]
    img_colorful_crop = img_colorful[region['y_min']:region['y_max'], region['x_min']:region['x_max'], :]

    crop_particles_idx_y = np.logical_and(gt_particles[:, 1] > region['y_min'], gt_particles[:, 1] < region['y_max'])
    crop_particles_idx_x = np.logical_and(gt_particles[:, 0] > region['x_min'], gt_particles[:, 0] < region['x_max'])
    crop_particles_idx = np.logical_and(crop_particles_idx_y, crop_particles_idx_x)
    crop_particles = gt_particles[crop_particles_idx]

    crop_particles[:,0] -= region['x_min']
    crop_particles[:,1] -= region['y_min']

    return img_crop, img_colorful_crop, crop_particles

def mirror_padding_image(img, boundary_size, GT_data):
    img_top_mirror = np.flip(img[:boundary_size['y'], :], 0)
    img_bottom_mirror = np.flip(img[-boundary_size['y']:, :], 0)
    tmp = np.vstack((img_top_mirror, img, img_bottom_mirror))

    img_left_mirror = np.flip(tmp[:, :boundary_size['x']], 1)
    img_right_mirror = np.flip(tmp[:, -boundary_size['x']:], 1)
    padded_img = np.hstack((img_left_mirror, tmp, img_right_mirror))

    GT_data[:,1] += boundary_size['y']
    GT_data[:,0] += boundary_size['x']

    return padded_img, GT_data

if __name__ == '__main__':
    GT_data = pd.read_csv('dataset/BAK1008L1_2020_07_02_11_56_18_AOSLO_788_V006_annotated_JLR_128_97_646_612.csv')
    GT_data = GT_data.to_numpy()
    img = cv2.imread('dataset/BAK1008L1_2020_07_02_11_56_18_AOSLO_788_V006_annotated_JLR_128_97_646_612.tiff', -1)
    img_colorful = cv2.imread('dataset/BAK1008L1_2020_07_02_11_56_18_AOSLO_788_V006_annotated_JLR_128_97_646_612.tiff')

    region = {
        'x_min': 500, # enumerate from zero !
        'x_max': 518,
        'y_min': 500,
        'y_max': 515,
    }

    img, img_colorful, GT_data = crop_image(img, img_colorful, GT_data, region)

    boundary_size = {
        'x': 10,
        'y': 5
    }
    mirror_padding_image(img, boundary_size, GT_data)
