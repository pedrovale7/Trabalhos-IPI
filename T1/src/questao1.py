import cv2, sys ,os
import numpy as np
from utils import abrir_img, duplicar, intercalar
import matplotlib.pyplot as plt

path = "T1/images/foto_perfil_gray.jpg"
matriz = abrir_img.open_img(path)

resultado_1 = intercalar.umquarto_img(matriz)
print(f'Tamanho Original :{matriz.shape}, e tamanho depois das funções :{resultado_1.shape}')
print("O tamanho diminui pela metade e a imagem aparentou igualzinha")

pasta = "T1/images/"
cv2.imwrite(f"{pasta}Resultado-1.jpg", resultado_1)

path2 = "T1/images/Foto_perfil.jpeg"

matriz2 = cv2.imread(path2, cv2.IMREAD_COLOR)
# print(matriz2.shape)

canal_B = intercalar.umquarto_img(matriz2[:,:,0])
canal_G = intercalar.umquarto_img(matriz2[:,:,1])
canal_R = intercalar.umquarto_img(matriz2[:,:,2])
# print(canal_B.shape)

resultado_2 = np.stack([canal_B, canal_G, canal_R], axis=2)
cv2.imwrite(f"{pasta}Resultado-2.jpg", resultado_2)

