import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import time
from pathlib import Path

import Train
from Agents.RandomAgent import RandomAgent
from Agents.LeftRightAgent import LeftRightAgent
from Agents.ListAgent import ListAgent
from Agents.DDQNAgent import DDQNAgent
from Agents.LoadSaveAgent import LoadSaveAgent
from ActionSpace import generate_action_space
from JumpKing import JKGame
from Constants import *

if __name__ == "__main__":
	#Game = JKGame()
	#Game.running()

	train = True

	path = "model_ddqn_episode"
	ddqn_state_dimension = 3
	action_space = generate_action_space(num_of_actions=10)

	if (train):
		t = Train.Train(DDQNAgent(	state_dim=ddqn_state_dimension,
						    		action_dim=len(action_space),
									is_training=train),
								action_space=action_space,
								csv_savepath= path + ".csv",
								agent_savepath= path + ".pth",
								)
	else:
		t = Train.Train(DDQNAgent(	state_dim=ddqn_state_dimension,
						    		action_dim=len(action_space),
									is_training=train),
								action_space=action_space,
								csv_savepath= path + ".csv",
								agent_loadpath= path + ".pth"
								)

	#t = Train.Train(RandomAgent(len(action_space)), action_space=action_space, csv_savepath="test.csv")

	t.run()