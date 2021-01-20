import unittest
import random
from lib.fps import FPS
import time

class TestFPS(unittest.TestCase):
	def test1(self):
		fps = FPS(50, 100)
		fps.count()
		while True:
			time.sleep(random.random()*10)
			fps.count()

