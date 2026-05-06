from PIL import Image
import numpy as np
 
def open_img(path):
    img = Image.open(path)
    matriz = np.array(img)
    return matriz