'''
Agente para comprobar la función load() y save() funcionan correctamente
'''

from Train import Agent

class LoadSaveAgent(Agent):

    def __init__(self):
        self.number = -1

    def load(self, path):
        with open(path, "r") as file:
            self.number = int(file.read().strip())

    def select_action(self, state):
        self.number += 1
        return 0 # placeholder
    
    def save(self, path):
        with open(path, "w") as file:
            file.write(str(self.number))


    