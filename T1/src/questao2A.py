import cv2
import numpy as np
path = "T1/images/Image1.pgm"
pasta = "T1/images/"
img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

kernel_8 = np.array([[-1, -1, -1],
                      [-1, 8, -1],
                      [-1, -1, -1]], dtype=np.float32)

laplaciano_8 = cv2.filter2D(img, cv2.CV_64F, kernel=kernel_8)
resultado_1 = img.astype(np.float64) + 0.2 *laplaciano_8  # Estou somando pq utilizei o centro positivo
resultado_1 = np.clip(resultado_1, 0, 255).astype(np.uint8)
cv2.imwrite(f"{pasta}Resultado_filtro_laplaciano.png", resultado_1)

gauss = cv2.GaussianBlur(img, (3,3), np.sqrt(0.5))
laplaciano_e_gauss = cv2.Laplacian(gauss, cv2.CV_64F, ksize=1)
resultado_2 = img.astype(np.float64) - laplaciano_e_gauss
resultado_2 = np.clip(resultado_2, 0, 255).astype(np.uint8)
cv2.imwrite(f'{pasta}Filtro_Gauss_Laplaciano.png', resultado_2)

gauss2 = cv2.GaussianBlur(img, (3,3), 1)
laplaciano_e_gauss2 = cv2.Laplacian(gauss2, cv2.CV_64F, ksize=1)
resultado_3 = img.astype(np.float64) - laplaciano_e_gauss2
resultado_3 = np.clip(resultado_3, 0, 255).astype(np.uint8)
cv2.imwrite(f'{pasta}Filtro_Gauss_Laplaciano2.png', resultado_3)

