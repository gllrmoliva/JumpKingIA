import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import time
from pathlib import Path

from DDQN import DDQN

import Train
from DumbAgent import DumbAgent
from RandomAgent import RandomAgent
from JumpKing import JKGame

#TODO: Esta función la podriamos generalizar para todos los agentes 
def train():
	'''Funcion para entrenar la IA'''
	action_dict = {
		0: 'right',
		1: 'left',
		2: 'right+space',
		3: 'left+space',
		# 4: 'space',
		# 5: 'idle',
	}
	agent = DDQN()
	env = JKGame(max_step=1000)
	num_episode = 100000

	for i in range(num_episode):
		done, state = env.reset()

		running_reward = 0
		while not done:
			action = agent.select_action(state)
			#print(action_dict[action])
			next_state, reward, done = env.step(action)

			running_reward += reward
			sign = 1 if done else 0
			agent.train(state, action, reward, next_state, sign)
			state = next_state
		print (f'episode: {i}, reward: {running_reward}')

			
if __name__ == "__main__":
	#Game = JKGame()
	#Game.running()

	#train()

	t = Train.Train(DumbAgent())
	t.run()