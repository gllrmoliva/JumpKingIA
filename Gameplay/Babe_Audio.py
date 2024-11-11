#!/usr/bin/env python
#
#
#
#

import pygame
import os
import re
import collections
from pathlib import Path

class Babe_Audio:

	def __init__(self):

		self.directory = Path("Audio")

		self.audio = collections.defaultdict()

		self._load_audio("Babe")

	def _load_audio(self, file):
		
		audio_directory = self.directory / file

		for audio in os.listdir(str(audio_directory)):

			a = pygame.mixer.Sound(str(audio_directory / audio))

			a.set_volume(0.5)

			self.audio[re.match(r"[^.]*", audio).group()] = a