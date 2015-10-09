__author__ = 'artemiibezguzikov'

import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('low-contrast.png')
img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

hist, bins = np.histogram(img.flatten(), 256, [0, 256])

cdf = hist.cumsum()

cdf_m = np.ma.masked_equal(cdf,0)
cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())


cdf = np.ma.filled(cdf_m, 0).astype('uint8')
equ = cv2.equalizeHist(img)
img2 = cdf[img]
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cl1 = clahe.apply(img)

res = np.hstack((img, equ))
res = np.vstack((res, np.hstack((img2, cl1))))
res = cv2.resize(res, (0,0), fx=0.5, fy=0.5)
cv2.namedWindow('frame')
cv2.imshow('frame', res)
cv2.waitKey(0)
cv2.destroyAllWindows()