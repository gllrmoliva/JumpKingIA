'''
    En este archivo se encuentran:
    1. Parametros del programa, modificables
    2. Constantes importantes, no modificables
'''


'''
    Parametros de depuración y ejecución
'''
DEBUG_LEVEL_MATRIX = False # Imprime la matriz del nivel en una ventana aparte ¡Ralentiza mucho el programa!
NO_INTERFACE = True # ¡Falta verificar que no da fallos! ¡Inseguro!
CSV_COOLDOWN = 10 # Cuantos pasos deben pasar para que volver a escribir en un .csv

'''
    Valores por defecto de los parametros de entrenamiento (Modificable)
'''
STEPS_PER_EPISODE = 1000
NUMBER_OF_EPISODES = 1
STEPS_PER_SECOND = -1

'''
    Constantes del modelo de entrenamiento (No modificable)
'''
# Parametros (o constante?) del espacio de acciones
# ¡No se esta usando en ningun lado! tampoco creo que sea necesario. Sirve más de documentación que otra cosa
ACTION_SPACE : dict[int, str] = {
0: 'right',
1: 'left',
2: 'right+space',
3: 'left+space',
4: 'space',   # No es util considerar esta acción, por ahora al menos.
5: 'idle',    # No es util considerar esta acción
}

'''
    Parametros de la representación matricial de las colisiones (Modificable)
'''
LEVEL_MATRIX_HORIZONTAL_SIZE = 48
LEVEL_MATRIX_VERTICAL_SIZE = 36

'''
    Constantes del tamaño de los niveles (No modificable)
'''
LEVEL_HORIZONTAL_SIZE = 480 # Cuanto mide el nivel horizontalmente, es una constante del repositorio original
LEVEL_VERTICAL_SIZE = 360 # Cuanto mide el nivel verticalmente, es una constante del repositorio original