import cv2
import numpy as np
path = "T1/images/moire.tif"
pasta = "T1/images/"

img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
def butterworth(img, n):
    h,w = img.shape
    F = np.fft.fftshift(np.fft.fft2(img))

    u,v = np.meshgrid(np.arange(w), np.arange(h))
    D = np.sqrt((u - w//2)**2 + (v - h//2)**2)

    d0 = 30
    H = 1/ (1 + (D/d0)**(2*n))
    G = F* H
    g = np.real(np.fft.ifft2(np.fft.ifftshift(G)))

    resultado = np.clip(g,0, 255).astype(np.uint8)
    return resultado

img_final = butterworth(img=img, n=4)
cv2.imwrite(f"{pasta}Butterworth.png", img_final)

def notch(img, points, n=4, padding=False):
    h,w = img.shape

    if padding:
        img_padding = np.zeros((2*h, 2*w), dtype=np.uint8)
        img_padding[0:h, 0:w] = img
        h, w = 2*h, 2*w

    F = np.fft.fftshift(np.fft.fft2(img))
    d0 = 10
    u,v = np.meshgrid(np.arange(w), np.arange(h))
    H = np.ones((h,w), dtype=np.float32)

    for u_k, v_k in points:

        D = np.sqrt((u - w//2 -u_k)**2 + (v - h//2 -v_k)**2)
        D_inv = np.sqrt((u - w//2 +u_k)**2 + (v - h//2 +v_k)**2)

        h_k = 1/ (1 + (d0 / D)**(2*n))
        h_inv = 1/ (1 + (d0 / D_inv)** (2*n))
        H *= h_k * h_inv

    G = F* H
    g = np.real(np.fft.ifft2(np.fft.ifftshift(G)))
    
    if padding == True:
        return np.clip(g[0:h, 0:w], 0, 255).astype(np.uint8)
    
    return  np.clip(g,0, 255).astype(np.uint8)


points = [(20,40), (-20,-40), (30,40), (-30, -40), (30,40), (-30, -40), (-27, 42), (27, -42)]
img_sempadding = notch(img, points)
cv2.imwrite(f"{pasta}Notch_sem_padding.png", img_sempadding)

h,w = img.shape
img_compadding = notch(img, points)
cv2.imwrite(f"{pasta}Notch_com_padding.png", img_compadding)


