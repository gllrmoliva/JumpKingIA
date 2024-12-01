'''
Agente que toma una acción aleatoria.
'''

from Train import Agent, State
from random import randint

class RandomAgent(Agent):

    def select_action(self, coded_state):
        return randint(0, 3)