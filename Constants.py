'''
    En este archivo se encuentran:
    1. Parametros del programa, modificables
    2. Constantes importantes, no modificables
'''


'''
    Parametros de depuración y ejecución
'''

NO_INTERFACE = False# Desactiva el salida gráfica (y en un futuro quizas la auditiva) para acelerar la simluación ¡Falta verificar que no da fallos! ¡Inseguro!
CSV_COOLDOWN = 10   # Cuantos pasos deben pasar para volver a escribir en el .csv
SAVE_COOLDOWN = 1 # Cada cuantos episodios se guardan copias del modelo

DEBUG_LEVEL_MATRIX = False # Imprime la matriz del nivel en una ventana aparte ¡Ralentiza mucho el programa!
DEBUG_OLD_COORDINATE_SYSTEM = False # Trabaja las coordenadas de un estado de la misma manera que el repositorio original

'''
    Parametros de entrenamiento (Modificable)
'''

STEPS_PER_EPISODE = 512   # Número de pasos máximo a simular en un episodio
NUMBER_OF_EPISODES = 50000 # Número de episodios a simular
STEPS_PER_SECOND = 60      # Cuántos pasos han de realizarse en un segundo, 'que tan rápido' es la simulación

                           # -1 : Desbloqueado, se realiza cuantos pasos permite el hardware

NO_LEVEL_MATRIX = True      # Desactiva la creación de la matriz del nivel ¡Por lo tanto un agente no puede utilizarla!

EPISODE_MAX_LEVEL = 5       # Si queremos mover al 'nivel final' del episodio a un nivel anterior en particular. Colocar un valor entre 0 y 42

ACTION_SPACE_SIZE = 10           # Número de acciones del espacio de acciones
WALKING_LENGTH = 10              # Por cuantos pasos camina, si es que escogió esa acción
INCLUDE_VERTICAL_SPACE = False   # ¿Incluir el salto vertical?

'''     ¡¡ OBSOLETO !!
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

'''
    Parametros de la representación matricial de las colisiones (Modificable)
'''
LEVEL_MATRIX_HORIZONTAL_SIZE = 48
LEVEL_MATRIX_VERTICAL_SIZE = 36
NEXT_LEVEL_MATRIX_HORIZONTAL_SIZE = 24
NEXT_LEVEL_MATRIX_VERTICAL_SIZE = 9

'''
    Constantes del repositorio original(No modificable)
'''
LEVEL_HORIZONTAL_SIZE = 480     # Cuanto mide el nivel horizontalmente, es una constante del repositorio original
LEVEL_VERTICAL_SIZE = 360       # Cuanto mide el nivel verticalmente, es una constante del repositorio original
JUMPCOUNT_MAX = 30              # Cuantos pasos seguidos se puede mantener el salto
GAME_MAX_LEVEL = 42             # Último nivel del juego (¡Se comienza a contar desde el 0!)
GAME_MAX_HEIGHT = 15480         # Altura máxima alcanzable