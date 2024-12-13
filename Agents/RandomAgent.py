'''
Agente que toma una acción aleatoria.
'''

from Train import Agent, State
from random import randint

class RandomAgent(Agent):

    def __init__(self, action_space_size):
        self.action_space_size = action_space_size

    def select_action(self, state):
        
        return randint(0, self.action_space_size - 1)