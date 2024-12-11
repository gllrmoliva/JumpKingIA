import numpy as np
import numpy.typing as npt
import cv2
from Constants import *
from math import floor, ceil

'''
Este archivo contiene principalmente la función de obtener la representación matricial de las colisiones de un nivel
Además contiene una serie de funciones adicionales auxiliares para esta representación
'''

if DEBUG_LEVEL_MATRIX:
    cv2.namedWindow("DEBUG_LEVEL_MATRIX", cv2.WINDOW_NORMAL)

# Función que obtiene una matriz que representa las colisiones del nivel
# Una celda vale 1 si hay una hitbox en la sección correspondiente del nivel. 0 en otro caso
# Recibe una instancia de JKGame y del numero del nivel en cuestion.
def get_level_matrix(game, level,
                     matrix_width = LEVEL_MATRIX_HORIZONTAL_SIZE, matrix_height = LEVEL_MATRIX_VERTICAL_SIZE,
                     position_rounding = floor, thickness_rounding = ceil,
                     debug = False):
    matrix = np.zeros((
                      matrix_height,
                      matrix_width),
                      dtype=np.uint8)
    
    platforms = game.levels.platforms.rectangles.levels[level]

    for p in platforms:

        x, y, w, h = area_to_matrix_area(p[0], p[1], p[2], p[3],
                                         matrix_width=matrix_width, matrix_height=matrix_height,
                                         position_rounding=position_rounding, thickness_rounding=thickness_rounding)
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

    if DEBUG_LEVEL_MATRIX and debug:
        frame = cv2.resize(matrix * 255, (400, 300))
        cv2.imshow("DEBUG_LEVEL_MATRIX", frame)
        cv2.waitKey(1) #ms
    
    return matrix

'''
Transforma coordenada x en la columna correspondiente de la matriz de colisiones
'''
def x_to_matrix_column(x, matrix_width = LEVEL_MATRIX_HORIZONTAL_SIZE, rounding = round):

    column = rounding(x * (matrix_width/LEVEL_HORIZONTAL_SIZE))
    if column < 0: column = 0
    elif column >= matrix_width: column = matrix_width - 1
    return column

'''
Transforma coordenada y en la fila correspondiente de la matriz de colisiones
'''
def y_to_matrix_row(y, matrix_height = LEVEL_MATRIX_VERTICAL_SIZE, rounding = round):

    row = rounding(y * (matrix_height/LEVEL_VERTICAL_SIZE))
    if row < 0: row = 0
    elif row >= matrix_height: row = matrix_height - 1
    return row

'''
Transforma coordenadas x, y en la celda correspondiente de la matriz de colisiones
Devuelve en la forma columna, fila
'''
def position_to_matrix_cell(x, y, matrix_width = LEVEL_MATRIX_HORIZONTAL_SIZE, matrix_height = LEVEL_MATRIX_VERTICAL_SIZE, rounding = round):

    return x_to_matrix_column(x, matrix_width=matrix_width, rounding=rounding), y_to_matrix_row(y, matrix_height=matrix_height, rounding=rounding)

'''
Transforma un area rectangular en el area correspondiente en la matriz de colisiones
Devuelve en la forma columna, fila, ancho, alto
'''
def area_to_matrix_area(x, y,
                        width, height, matrix_width = LEVEL_MATRIX_HORIZONTAL_SIZE, matrix_height = LEVEL_MATRIX_VERTICAL_SIZE,
                        position_rounding = floor, thickness_rounding = ceil):

    column, row = position_to_matrix_cell(x, y, matrix_width=matrix_width, matrix_height=matrix_height, rounding=position_rounding)

    output_width = thickness_rounding(width * (matrix_width/LEVEL_HORIZONTAL_SIZE))
    if column + output_width >= matrix_width: output_width = matrix_width - 1 - column

    output_height = thickness_rounding(height * (matrix_height/LEVEL_VERTICAL_SIZE))
    if row + output_height >= matrix_height: output_height = matrix_height - 1 - row
    
    if output_width == 0:
        column += -1
        output_width += 1
    if output_height == 0:
        row += -1
        output_height += 1

    return column, row, output_width, output_height