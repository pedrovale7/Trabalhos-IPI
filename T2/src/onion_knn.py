# O número de cluster vai ser a qtd de tons de cores na imagem?
# Agrupando grupos de acordo com as cores dos elementos

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

path = "T2/images"
img = cv.imread(f'{path}/onion.jpg')
h, w, c = img.shape

img = img.reshape((-1,3)).astype(np.float32)
print(img.shape)

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)

# compactness_list = []
# for k in range(3,10):
#     compactness, labels, centers = cv.kmeans(img, k, None,
#                                              criteria, 10,
#                                              cv.KMEANS_RANDOM_CENTERS)
#     compactness_list.append(compactness)
# print(compactness_list)
# plt.plot(range(3,10), compactness_list)
# plt.savefig(f'{path}/MelhorK.png')
# plt.show()


compactness, labels, centers = cv.kmeans(img, 5, None,
                                             criteria, 10,
                                             cv.KMEANS_RANDOM_CENTERS)

centers = np.uint8(centers)
img_segmentada = centers[labels.flatten()]
img_segmentada = img_segmentada.reshape((h,w,c))

cv.imwrite(f'{path}/img_segmentada.png', img_segmentada)
