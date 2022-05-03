import cv2
import pandas as pd

data = pd.read_csv('dataset/BAK1008L1_2020_07_02_11_56_18_AOSLO_788_V006_annotated_JLR_128_97_646_612.csv')
img = cv2.imread('dataset/BAK1008L1_2020_07_02_11_56_18_AOSLO_788_V006_annotated_JLR_128_97_646_612.tiff')

cv2.imshow('origin img', img)
cv2.waitKey()

image = img.copy()
for index, row in data.iterrows():
    cv2.circle(image, (round(row[0]),round(row[1])), radius=1, color=(0, 255, 255), thickness=-1)

cv2.imshow('img with centers', image)
cv2.waitKey()
