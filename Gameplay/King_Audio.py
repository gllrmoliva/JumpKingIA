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

class King_Audio:

	def __init__(self):

		self.directory = str(Path("Assets/Audio/King"))

		self.audio = collections.defaultdict()

		self.audio["Land"] = self._load_audio("Land")

		self.audio["Ice"] = self._load_audio("Ice")

		self.audio["Snow"] = self._load_audio("Snow")

	def _load_audio(self, file):

		audio_dict = collections.defaultdict()

		for audio in os.listdir(str(Path(f"{self.directory}/{file}"))):

			a = pygame.mixer.Sound(str(Path(f"{self.directory}/{file}/{audio}")))

			a.set_volume(1.0)

			audio_dict[re.match(r"[^.]*", audio).group()] = a

		return audio_dict