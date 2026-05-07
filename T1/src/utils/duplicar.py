import numpy as np

def duplicar_linha(matriz):
    h, w = matriz.shape
    nova_matriz = np.zeros((h*2, w))

    for i in range(h):
        nova_matriz[2*i] = matriz[i]

        if i < h-1:
            nova_matriz[2*i +1] = (matriz[i].astype(float) + matriz[i+1].astype(float))/2
        else:
            nova_matriz[2*i+1] = matriz[i] # a ultima linha copia a de cima
    
    nova_matriz = nova_matriz.astype(np.int8)
    return nova_matriz

def duplicar_coluna(matriz):
    h, w = matriz.shape
    nova_matriz = np.zeros((h, w*2))

    for j in range(w):
        nova_matriz[2*j] = matriz[j]
        
        if j < w-1:
            nova_matriz[2*j +1] = (matriz[j].astype(float) + matriz[j+1].astype(float))/2
        else:
            nova_matriz[2*j+1] = matriz[j] # a ultima linha copia a de cima
    
    nova_matriz = nova_matriz.astype(np.int8)
    return nova_matriz

def duplicar(matriz):
    linha_duplicada = duplicar_linha(matriz)
    duplicar_ambas = duplicar_coluna(linha_duplicada) 

    return duplicar_ambas