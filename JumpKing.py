#!/usr/env/bin python
#   
# Game Screen
# 

'''
Contiene la clase JKGame la cual sirve para 'instanciar' la aplicación
En este archivo antes estaba el main() y la función train(), las movi a main.py ¡Daba problemas de importación circular!
'''

import pygame 
import sys
import os
import inspect
import pickle
import numpy as np
from Gameplay.environment import Environment
from Gameplay.spritesheet import SpriteSheet
from Gameplay.Background import Backgrounds
from Gameplay.King import King
from Gameplay.Babe import Babe
from Gameplay.Level import Levels
from Gameplay.Menu import Menus

from Gameplay.Start import Start
from pathlib import Path

from Constants import LEVEL_VERTICAL_SIZE, NO_INTERFACE, DEBUG_OLD_COORDINATE_SYSTEM

# Keyword
FPS_UNLOCKED = -1

class JKGame:
	""" Overall class to manga game aspects """

	"""
	Función auxiliar para actualizar variables relacionadas a la altura
	"""
	def _update_heights(self):
		self.height = (LEVEL_VERTICAL_SIZE - self.king.rect_y) + self.king.levels.current_level * LEVEL_VERTICAL_SIZE
		if self.height > self.max_height:
			self.max_height = self.height
		if self.height > self.max_height_last_step:
			self.max_height_last_step = self.height
        
	"""
	Constructor para instanciar la aplicación.
	Parametros:
		steps_per_episode: Cantidad de 'pasos' antes de terminar un episodio
		steps_per_seconds: Cantidad de 'pasos' de la simulación que se realizan en un segundo
			-1: Desbloqueado, ejecuta al mayor ritmo que puede.
	"""
	def __init__(self, steps_per_seconds):

		# Variables nuevas / modificadas

		self.height = 0	# Altura total actual del King
		self.max_height = 0 # Altura total máxima que ha alcanzado el King
		self.max_height_last_step = 0 # Altura total máxima que alcanzó el King en el último paso

		#

		pygame.init()

		self.environment = Environment()

		self.clock = pygame.time.Clock()

		if steps_per_seconds > 0:
			self.fps = steps_per_seconds
		elif steps_per_seconds == -1:
			self.fps = FPS_UNLOCKED
		else:
			raise ValueError("Invalid value of steps_per_seconds parameter")
		
		self.screen = pygame.display.set_mode((int(os.environ.get("screen_width")) * int(os.environ.get("window_scale")), int(os.environ.get("screen_height")) * int(os.environ.get("window_scale"))), pygame.HWSURFACE|pygame.DOUBLEBUF)#|pygame.SRCALPHA)
		
		if not NO_INTERFACE:
 
			self.bg_color = (0, 0, 0)
			self.game_screen = pygame.Surface((int(os.environ.get("screen_width")), int(os.environ.get("screen_height"))), pygame.HWSURFACE|pygame.DOUBLEBUF)#|pygame.SRCALPHA)
			self.game_screen_x = 0
			pygame.display.set_icon(pygame.image.load(str(Path("Assets/images/sheets/JumpKingIcon.ico"))))
		
		else:

			self.game_screen = None
		
		self.levels = Levels(self.game_screen)

		self.king = King(self.game_screen, self.levels)

		self.babe = Babe(self.game_screen, self.levels)

		self.menus = Menus(self.game_screen, self.levels, self.king)

		self.start = Start(self.game_screen, self.menus)

		self.visited = {}

		self._update_heights()

		pygame.display.set_caption('Jump King At Home XD')

	def reset(self):
		'''Método para reiniciar el juego'''

		self.height = 0	# Altura total actual del King
		self.max_height = 0 # Altura total máxima que ha alcanzado el King
		self.max_height_last_step = 0 # Altura total máxima que alcanzó el King en el último paso

		self.king.reset()
		self.levels.reset()
		os.environ["start"] = "1"
		os.environ["gaming"] = "1"
		os.environ["pause"] = ""
		os.environ["active"] = "1"
		os.environ["attempt"] = str(int(os.environ.get("attempt")) + 1)
		os.environ["session"] = "0"

		self.visited = {}
		self.visited[(self.king.levels.current_level, self.king.y)] = 1

		self._update_heights()

		return

	def move_available(self):
		'''Metodo para conseguir movimientos disponibles'''
		available = not self.king.isFalling \
					and not self.king.levels.ending \
					and (not self.king.isSplat or self.king.splatCount > self.king.splatDuration)
		return available

	def step(self, action):
		'''Metodo para realizar un paso en el juego,
		¡Sólo es utilizado para cuando se ejecuta el juego con un agente! En caso de ser un jugador humano se ejectua running()'''

		#old_level = self.king.levels.current_level
		#old_y = self.king.y
		##old_y = (self.king.levels.max_level - self.king.levels.current_level) * 360 + self.king.y

		self.max_height_last_step = self.height

		while True:

			if self.fps != FPS_UNLOCKED: # fps desbloqueados
				self.clock.tick(self.fps)

			self._check_events()

			if not os.environ["pause"]:
				if not self.move_available():
					action = None
				self._update_gamestuff(action=action)

			if not NO_INTERFACE:
				self._update_gamescreen()
				self._update_guistuff()
				self._update_audio()
				pygame.display.update()
			
			self._update_heights()

			if self.move_available():

				##################################################################################################
				# Define the reward from environment                                                             #
				##################################################################################################
				'''
				if self.king.levels.current_level > old_level or (self.king.levels.current_level == old_level and self.king.y < old_y):
					reward = 0
				else:
					self.visited[(self.king.levels.current_level, self.king.y)] = self.visited.get((self.king.levels.current_level, self.king.y), 0) + 1
					if self.visited[(self.king.levels.current_level, self.king.y)] < self.visited[(old_level, old_y)]:
						self.visited[(self.king.levels.current_level, self.king.y)] = self.visited[(old_level, old_y)] + 1

					reward = -self.visited[(self.king.levels.current_level, self.king.y)]
				'''
				####################################################################################################
				
				return

	def running(self):
		"""
		play game with keyboard
		:return:
		"""
		self.reset()
		while True:
			#state = [self.king.levels.current_level, self.king.x, self.king.y, self.king.jumpCount]
			#print(state)
			if self.fps != FPS_UNLOCKED: # fps desbloqueados
				self.clock.tick(self.fps)
			self._check_events()
			if not os.environ["pause"]:
				self._update_gamestuff()

			if not NO_INTERFACE:
				self._update_gamescreen()
				self._update_guistuff()
				self._update_audio()
				pygame.display.update()

	def _check_events(self):
		'''Metodo para verificar eventos'''

		for event in pygame.event.get():

			if event.type == pygame.QUIT:

				self.environment.save()

				self.menus.save()

				sys.exit()

			if event.type == pygame.KEYDOWN:

				self.menus.check_events(event)

				if event.key == pygame.K_c:

					if os.environ["mode"] == "creative":

						os.environ["mode"] = "normal"

					else:

						os.environ["mode"] = "creative"
					
			if event.type == pygame.VIDEORESIZE:

				self._resize_screen(event.w, event.h)

	def _update_gamestuff(self, action=None):
		'''Metodo para actualizar el juego'''

		self.levels.update_levels(self.king, self.babe, agentCommand=action)

	def _update_guistuff(self):
		'''Metodo para actualizar la interfaz de usuario'''

		if self.menus.current_menu:

			self.menus.update()

		if not os.environ["gaming"]:

			self.start.update()

	def _update_gamescreen(self):
		'''Metodo para actualizar la pantalla del juego'''

		pygame.display.set_caption(f"Jump King At Home XD - {self.clock.get_fps():.2f} FPS")

		self.game_screen.fill(self.bg_color)

		if os.environ["gaming"]:

			self.levels.blit1()

		if os.environ["active"]:

			self.king.blitme()

		if os.environ["gaming"]:

			self.babe.blitme()

		if os.environ["gaming"]:

			self.levels.blit2()

		if os.environ["gaming"]:

			self._shake_screen()

		if not os.environ["gaming"]:

			self.start.blitme()

		self.menus.blitme()

		self.screen.blit(pygame.transform.scale(self.game_screen, self.screen.get_size()), (self.game_screen_x, 0))

	def _resize_screen(self, w, h):
		'''Metodo para redimensionar la pantalla'''

		self.screen = pygame.display.set_mode((w, h), pygame.HWSURFACE|pygame.DOUBLEBUF|pygame.SRCALPHA)

	def _shake_screen(self):
		'''Metodo para agitar la pantalla'''

		try:

			if self.levels.levels[self.levels.current_level].shake:

				if self.levels.shake_var <= 150:

					self.game_screen_x = 0

				elif self.levels.shake_var // 8 % 2 == 1:

					self.game_screen_x = -1

				elif self.levels.shake_var // 8 % 2 == 0:

					self.game_screen_x = 1

			if self.levels.shake_var > 260:

				self.levels.shake_var = 0

			self.levels.shake_var += 1

		except Exception as e:

			print("SHAKE ERROR: ", e)

	def _update_audio(self):
		'''Metodo para actualizar el audio'''

		for channel in range(pygame.mixer.get_num_channels()):

			if not os.environ["music"]:

				if channel in range(0, 2):

					pygame.mixer.Channel(channel).set_volume(0)

					continue

			if not os.environ["ambience"]:

				if channel in range(2, 7):

					pygame.mixer.Channel(channel).set_volume(0)

					continue

			if not os.environ["sfx"]:

				if channel in range(7, 16):

					pygame.mixer.Channel(channel).set_volume(0)

					continue

			pygame.mixer.Channel(channel).set_volume(float(os.environ.get("volume")))