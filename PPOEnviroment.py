from JumpKing import JKGame
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from Constants import *
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
import ActionSpace
from Train import State, Environment
import os
import tensorflow as tf

class JumpKingEnv(gym.Env):

    def __init__(self, steps_per_episode=STEPS_PER_EPISODE, steps_per_second=-1, total_rewards=[], max_height=[]):
        super().__init__()
        self.env = Environment(steps_per_episode, steps_per_second)
        self.steps_per_episode = steps_per_episode

        # Generar el espacio de acciones dinámico
        self.action_space_dict = ActionSpace.generate_action_space(num_of_actions=10)
        self.action_space = spaces.Discrete(len(self.action_space_dict))  # Espacio discreto basado en el tamaño
        self.reward = 0
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0]),  # Valores mínimos de level, x, y, height, etc.
            high=np.array([GAME_MAX_LEVEL, LEVEL_HORIZONTAL_SIZE,
                           GAME_MAX_HEIGHT, GAME_MAX_HEIGHT]),  # Valores máximos
            dtype=np.float32
        )
        self.total_rewards = total_rewards
        self.max_heights = max_height

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

        self.reward = 0
        # Recompensa por altura
        delta_altura = self.next_state.height_normalized - self.state.height_normalized
        if delta_altura > 0:
            self.reward += delta_altura * 10
        else:
            self.reward += delta_altura * 10
        
        # Penalización por quedarse en el mismo lugar
        if self.next_state.x_normalized == self.state.x_normalized and self.next_state.level == self.state.level:
            self.reward -= 3

        if self.next_state.height_normalized == self.state.height_normalized:
            self.reward -= 4

        if self.next_state.max_height_normalized == self.state.max_height_normalized:
            self.reward -= 4
        
        if self.state.y_normalized == self.next_state.y_normalized:
            self.reward -= 4

        if self.next_state.level > self.state.level:
            self.reward += 100
        elif self.next_state.level < self.state.level:
            self.reward -= 100

        done = self.next_state.done

        # Premio adicional por alcanzar una altura nueva
        if self.next_state.max_height_normalized > self.state.max_height_normalized:
            self.reward += 3

        if self.next_state.win:
            self.reward += 100  # Recompensa grande por ganar

        if self.next_state.done:
            self.reward += 100 if self.next_state.max_height_normalized > self.state.max_height_normalized else -100

        self.reward += self.state.height_normalized

        observation = self._state_to_observation(self.next_state)

        self.max_heights.append(self.state.max_height)
        self.total_rewards.append(self.reward)
        return observation, self.reward, done, False, {}
    
    def _state_to_observation(self, state: State):
        return np.array([
            state.level_normalized,
            state.x_normalized,
            state.height_normalized,
            state.max_height_normalized
        ], dtype=np.float32)

    def _action_to_tuple(self, action):
        # Mapeo del espacio discreto a tupla
        action_map = ActionSpace.generate_action_space(num_of_actions=10)
        return action_map[action]

    def render(self, mode="human"):
        if mode == "human":
            self.env.game.render()

if __name__ == "__main__":
    
    models_dir = f"PPO/models"
    logdir = f"PPO/logs"
    writer = tf.summary.create_file_writer(logdir)
    total_rewards = []
    max_height = []

    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    if not os.path.exists(logdir):
        os.makedirs(logdir)
    env = JumpKingEnv(total_rewards=total_rewards, max_height=max_height)
    # Verificar si existe un modelo previamente entrenado
    if os.path.exists(models_dir) and len(os.listdir(models_dir)) > 0:
        # Obtener el modelo más reciente guardado
        saved_models = sorted(os.listdir(models_dir), key=lambda x: int(x.split("_")[0]))
        latest_model_path = os.path.join(models_dir, saved_models[-1])
        print(f"Cargando modelo previamente entrenado: {latest_model_path}")
        model = PPO.load(latest_model_path, env=env, tensorboard_log=logdir, device="cpu")
    else:
        print("No se encontró ningún modelo entrenado. Creando uno nuevo.")
        model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=logdir, device="cpu")

    check_env(env)  # Asegúrate de que el entorno cumple con la API

    for i in range(NUMBER_OF_EPISODES):
        model.learn(total_timesteps=STEPS_PER_EPISODE)
        
        with writer.as_default():
            reward = env.reward  # Recompensa acumulada o actual del episodio
            print(f"Episodio {i}: {reward}")
            tf.summary.scalar("Reward", total_rewards[-1], step=i)
            tf.summary.scalar("Episode", i, step=i)
            tf.summary.scalar("Altura", max_height[-1], step=i)

        model.save(f"{models_dir}/{(STEPS_PER_EPISODE*i)+343000}_Steps")

    """ Para poder ver las graficas, en la terminal ejecutar tensorboard --logdir=PPO/logs """
    writer.close()
    env.close()