import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
path = "T2/images/brain.jpg"
img = cv.imread(path, cv.IMREAD_GRAYSCALE)
# print(img.shape)
# cv.imshow("Janela", img)
# cv.waitKey(0)

gauss_filter = cv.GaussianBlur(img , (5,5), 1)
cv.imwrite("T2/images/Gauss_Img.jpeg", gauss_filter)

median_filter = cv.medianBlur(img, 5)
cv.imwrite("T2/images/Median_Img.jpeg", median_filter)

histogram = cv.calcHist([median_filter], [0], None, [256], [0,256])
plt.hist(histogram, 256, [0,256])
plt.title("Histograma da Imagem")
plt.savefig("T2/images/Histograma.jpeg")

_, img_bin = cv.threshold(median_filter, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
cv.imwrite("T2/images/Img_bin.jpeg", img_bin)

disk7 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7,7))
open_img = cv.morphologyEx(img_bin, cv.MORPH_OPEN, disk7)
open_close_img = cv.morphologyEx(open_img, cv.MORPH_CLOSE, disk7)

num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(open_close_img, connectivity=8)

index = np.argsort(stats[1:, cv.CC_STAT_AREA]) + 1

tumor_isolado = np.zeros_like(open_close_img)
tumor_isolado[labels == index[-2]] = 255
cv.imwrite("T2/images/Tumor_Isolado.png", tumor_isolado)
