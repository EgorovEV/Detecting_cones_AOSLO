import cv2
import pandas as pd
import matplotlib.pyplot as plt
from utils import GT_enumerate_from_zero

data = pd.read_csv('dataset/BAK1008L1_2020_07_02_11_56_18_AOSLO_788_V006_annotated_JLR_128_97_646_612.csv')
img = cv2.imread('dataset/BAK1008L1_2020_07_02_11_56_18_AOSLO_788_V006_annotated_JLR_128_97_646_612.tiff')

data = GT_enumerate_from_zero(data)

plt.imshow(img)
plt.show()

image = img.copy()
for index, row in data.iterrows():
    # image, x_coord, y_coord -- from docs
    cv2.circle(image, (round(row[0]),round(row[1])), radius=0, color=(255, 0, 0), thickness=-1)

plt.imshow(image)
plt.show()
