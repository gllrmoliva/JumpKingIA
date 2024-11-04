import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class NETWORK(torch.nn.Module):
	def __init__(self, input_dim: int, output_dim: int, hidden_dim: int) -> None:
		"""DQN Network example
        Args:
            input_dim (int): `state` dimension.
                `state` is 2-D tensor of shape (n, input_dim)
            output_dim (int): Number of actions.
                Q_value is 2-D tensor of shape (n, output_dim)
            hidden_dim (int): Hidden dimension in fc layer
        """
		super(NETWORK, self).__init__()

		self.layer1 = torch.nn.Sequential(
			torch.nn.Linear(input_dim, hidden_dim),
			torch.nn.ReLU()
		)

		self.layer2 = torch.nn.Sequential(
			torch.nn.Linear(hidden_dim, hidden_dim),
			torch.nn.ReLU()
		)

		self.final = torch.nn.Linear(hidden_dim, output_dim)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		"""Returns a Q_value
        Args:
            x (torch.Tensor): `State` 2-D tensor of shape (n, input_dim)
        Returns:
            torch.Tensor: Q_value, 2-D tensor of shape (n, output_dim)
        """
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.final(x)

		return x


class DDQN(object):
	def __init__(
			self
	):
		self.target_net = NETWORK(4, 4, 32)
		self.eval_net = NETWORK(4, 4, 32)

		self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=0.001)
		self.criterion = nn.MSELoss()

		self.memory_counter = 0
		self.memory_size = 50000
		self.memory = np.zeros((self.memory_size, 11))

		self.epsilon = 1.0
		self.epsilon_decay = 0.95
		self.alpha = 0.99

		self.batch_size = 64
		self.episode_counter = 0

		self.target_net.load_state_dict(self.eval_net.state_dict())

	def memory_store(self, s0, a0, r, s1, sign):
		transition = np.concatenate((s0, [a0, r], s1, [sign]))
		index = self.memory_counter % self.memory_size
		self.memory[index, :] = transition
		self.memory_counter += 1

	def select_action(self, states: np.ndarray) -> int:
		state = torch.unsqueeze(torch.tensor(states).float(), 0)
		if np.random.uniform() > self.epsilon:
			logit = self.eval_net(state)
			action = torch.argmax(logit, 1).item()
		else:
			action = int(np.random.choice(4, 1))

		return action

	def policy(self, states: np.ndarray) -> int:
		state = torch.unsqueeze(torch.tensor(states).float(), 0)
		logit = self.eval_net(state)
		action = torch.argmax(logit, 1).item()

		return action

	def train(self, s0, a0, r, s1, sign):
		if sign == 1:
			if self.episode_counter % 2 == 0:
				self.target_net.load_state_dict(self.eval_net.state_dict())
			self.episode_counter += 1

		self.memory_store(s0, a0, r, s1, sign)
		self.epsilon = np.clip(self.epsilon * self.epsilon_decay, a_min=0.01, a_max=None)

		# select batch sample
		if self.memory_counter > self.memory_size:
			batch_index = np.random.choice(self.memory_size, size=self.batch_size)
		else:
			batch_index = np.random.choice(self.memory_counter, size=self.batch_size)

		batch_memory = self.memory[batch_index]
		batch_s0 = torch.tensor(batch_memory[:, :4]).float()
		batch_a0 = torch.tensor(batch_memory[:, 4:5]).long()
		batch_r = torch.tensor(batch_memory[:, 5:6]).float()
		batch_s1 = torch.tensor(batch_memory[:, 6:10]).float()
		batch_sign = torch.tensor(batch_memory[:, 10:11]).long()

		q_eval = self.eval_net(batch_s0).gather(1, batch_a0)

		with torch.no_grad():
			maxAction = torch.argmax(self.eval_net(batch_s1), 1, keepdim=True)
			q_target = batch_r + (1 - batch_sign) * self.alpha * self.target_net(batch_s1).gather(1, maxAction)

		loss = self.criterion(q_eval, q_target)

		# backward
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()