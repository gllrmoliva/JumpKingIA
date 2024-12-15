from JumpKing import JKGame
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from Constants import *
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
import ActionSpace
from Train import State, Environment
import time
import os
import tensorflow as tf

class JumpKingEnv(gym.Env):

    def __init__(self, steps_per_episode=STEPS_PER_EPISODE, steps_per_second=-1, total_rewards=[]):
        super().__init__()
        self.env = Environment(steps_per_episode, steps_per_second)
        self.steps_per_episode = steps_per_episode

        # Generar el espacio de acciones dinámico
        self.action_space_dict = ActionSpace.generate_action_space(num_of_actions=10)
        self.action_space = spaces.Discrete(len(self.action_space_dict))  # Espacio discreto basado en el tamaño
        self.reward = 0
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0]),  # Valores mínimos de level, x, y, height, etc.
            high=np.array([GAME_MAX_LEVEL, LEVEL_HORIZONTAL_SIZE, LEVEL_VERTICAL_SIZE, 
                           GAME_MAX_HEIGHT, GAME_MAX_HEIGHT, JUMPCOUNT_MAX]),  # Valores máximos
            dtype=np.float32
        )
        self.total_rewards = total_rewards
        self.max_height = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        state = self.env.reset()
        observation = self._state_to_observation(state)
        self.max_height = 0
        return observation, {}
    
    def step(self, action):
        action_tuple = self._action_to_tuple(action)
        self.state = State.get_state_from_env(self.env)
        self.next_state = self.env.step(action_tuple)

        # Recompensa por altura
        delta_altura = self.next_state.height - self.state.height
        if delta_altura > 0:
            self.reward = delta_altura 
        else:
            self.reward = delta_altura * 1.5
        
        # Penalización por quedarse en el mismo lugar
        self.reward -= 0.5 if self.next_state.x == self.state.x else 0

        self.reward += 100 if self.next_state.level > self.state.level else 0
        done = self.next_state.done

        observation = self._state_to_observation(self.next_state)
        self.max_height = self.max_height 
        self.total_rewards.append(self.reward)
        return observation, self.reward, done, False, {}
    
    def _state_to_observation(self, state: State):
        return np.array([
            state.level,
            state.x,
            state.y,
            state.height,
            state.max_height,
            state.jumpCount
        ], dtype=np.float32)

    def _action_to_tuple(self, action):
        # Mapeo del espacio discreto a tupla
        action_map = ActionSpace.generate_action_space(num_of_actions=10)
        return action_map[action]

    def render(self, mode="human"):
        if mode == "human":
            self.env.game.render()


models_dir = f"PPO/models"
logdir = f"PPO/logs/PPO-{int(time.time())}"
writer = tf.summary.create_file_writer(logdir)
total_rewards = []

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)
env = JumpKingEnv(total_rewards=total_rewards)
# Verificar si existe un modelo previamente entrenado
if os.path.exists(models_dir) and len(os.listdir(models_dir)) > 0:
    # Obtener el modelo más reciente guardado
    saved_models = sorted(os.listdir(models_dir), key=lambda x: int(x.split("_")[0]))
    latest_model_path = os.path.join(models_dir, saved_models[-1])
    print(f"Cargando modelo previamente entrenado: {latest_model_path}")
    model = PPO.load(latest_model_path, env=env, tensorboard_log=logdir)
else:
    print("No se encontró ningún modelo entrenado. Creando uno nuevo.")
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=logdir)

check_env(env)  # Asegúrate de que el entorno cumple con la API
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=logdir)

for i in range(NUMBER_OF_EPISODES):
    model.learn(total_timesteps=STEPS_PER_EPISODE)
    
    with writer.as_default():
        reward = env.reward  # Recompensa acumulada o actual del episodio
        tf.summary.scalar("Reward", total_rewards[-1], step=i)
        tf.summary.scalar("Episode", i, step=i)
        tf.summary.scalar("Altura", env.max_height, step=i)

    model.save(f"{models_dir}/{STEPS_PER_EPISODE*i}_Steps")

""" Para poder ver las graficas, en la terminal ejecutar tensorboard --logdir=PPO/logs """
writer.close()
env.close()