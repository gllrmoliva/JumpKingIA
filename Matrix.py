import numpy as np
import numpy.typing as npt
import cv2
from Constants import *

'''
Este archivo contiene principalmente la función de obtener la representación matricial de las colisiones de un nivel
Además contiene una serie de funciones adicionales auxiliares para esta representación
'''

if DEBUG_LEVEL_MATRIX:
    cv2.namedWindow("DEBUG_LEVEL_MATRIX", cv2.WINDOW_NORMAL)

# Función que obtiene una matriz que representa las colisiones del nivel
# Una celda vale 1 si hay una hitbox en la sección correspondiente del nivel. 0 en otro caso
# Recibe una instancia de Environment y del numero del nivel en cuestion.
def get_level_matrix(env, level):
    matrix = np.zeros((
                      LEVEL_MATRIX_VERTICAL_SIZE,
                      LEVEL_MATRIX_HORIZONTAL_SIZE),
                      dtype=np.uint8)
    
    platforms = env.game.levels.platforms.rectangles.levels[level]

    for p in platforms:

        x, y, w, h = area_to_matrix_area(p[0], p[1], p[2], p[3])
        slope = p[4]

        if slope == 0:
            matrix[y:y+h, x:x+w] = 1

        elif slope == (1, 1):
            triangle_mask = np.tril(np.ones((h, w)))
            reflected_triangle_mask = triangle_mask[:, ::-1]
            matrix[y:y+h, x:x+w] = reflected_triangle_mask 

        elif slope == (-1, 1):
            triangle_mask = np.tril(np.ones((h, w)))
            matrix[y:y+h, x:x+w] = triangle_mask 

        elif slope == (1, -1):
            triangle_mask = np.triu(np.ones((h, w)))
            reflected_triangle_mask = triangle_mask[:, ::-1]
            matrix[y:y+h, x:x+w] = reflected_triangle_mask 

        elif slope == (-1, -1):
            triangle_mask = np.triu(np.ones((h, w)))
            matrix[y:y+h, x:x+w] = triangle_mask 

    if DEBUG_LEVEL_MATRIX:
        frame = cv2.resize(matrix * 255, (400, 300))
        cv2.imshow("DEBUG_LEVEL_MATRIX", frame)
        cv2.waitKey(1) #ms
    
    return matrix

'''
Transforma coordenada x en la columna correspondiente de la matriz de colisiones
'''
def x_to_matrix_column(x):

    column = round(x * (LEVEL_MATRIX_HORIZONTAL_SIZE/LEVEL_HORIZONTAL_SIZE))
    if column < 0: column = 0
    elif column >= LEVEL_MATRIX_HORIZONTAL_SIZE: column = LEVEL_MATRIX_HORIZONTAL_SIZE - 1
    return column

'''
Transforma coordenada y en la fila correspondiente de la matriz de colisiones
'''
def y_to_matrix_row(y):

    row = round(y * (LEVEL_MATRIX_VERTICAL_SIZE/LEVEL_VERTICAL_SIZE))
    if row < 0: row = 0
    elif row >= LEVEL_MATRIX_VERTICAL_SIZE: row = LEVEL_MATRIX_VERTICAL_SIZE - 1
    return row

'''
Transforma coordenadas x, y en la celda correspondiente de la matriz de colisiones
Devuelve en la forma columna, fila
'''
def position_to_matrix_cell(x, y):

    return x_to_matrix_column(x), y_to_matrix_row(y)

'''
Transforma un area rectangular en el area correspondiente en la matriz de colisiones
Devuelve en la forma columna, fila, ancho, alto
'''
def area_to_matrix_area(x, y, width, height):

    column, row = position_to_matrix_cell(x, y)

    matrix_width = round(width * (LEVEL_MATRIX_HORIZONTAL_SIZE/LEVEL_HORIZONTAL_SIZE))
    if column + matrix_width >= LEVEL_MATRIX_HORIZONTAL_SIZE: matrix_width = LEVEL_MATRIX_HORIZONTAL_SIZE - 1 - column

    matrix_height = round(height * (LEVEL_MATRIX_VERTICAL_SIZE/LEVEL_VERTICAL_SIZE))
    if row + matrix_height >= LEVEL_MATRIX_VERTICAL_SIZE: matrix_height = LEVEL_MATRIX_VERTICAL_SIZE - 1 - row

    return column, row, matrix_width, matrix_height