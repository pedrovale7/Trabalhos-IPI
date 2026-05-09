import cv2
import numpy as np
path = "T1/images/Image1.pgm"
pasta = "T1/images/"

img = cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
h, w = img.shape

F = np.fft.fftshift(np.fft.fft2(img))
u,v = np.meshgrid(np.arange(w), np.arange(h))
D = (u - (w//2) **2) + (v - (h//2)**2) # vetor distancia

D0 = 100
H = (D > D0**2).astype(np.float32)

H_gauss = np.exp(-D / (2* D**2))
H_gauss = 1 - H_gauss

G = F * H * H_gauss
g = np.real(np.fft.ifft2(np.fft.ifftshift(G)))

k = 1
resultado = cv2.addWeighted(img.astype(np.float32), 1.0, g, k, 0, dtype=cv2.CV_32F)
resultado = np.clip(resultado, 0, 255).astype(np.uint8)
cv2.imwrite(f"{pasta}Passa_Alta_Gauss2.5.png", resultado)