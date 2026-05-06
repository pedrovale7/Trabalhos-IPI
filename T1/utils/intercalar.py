def intercalar_linha(matriz):
    matriz_reduzida = matriz[::2, : ]
    return matriz_reduzida

def intercalar_coluna(matriz):
    matriz_reduzida = matriz[:, ::2]
    return matriz_reduzida
 
def umquarto_img (matriz):
    matriz_reduzida = intercalar_linha(matriz)
    umquarto_matriz = intercalar_coluna(matriz_reduzida)
    return umquarto_matriz
