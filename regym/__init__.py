from multiprocessing.managers import SyncManager

class CustomManager(SyncManager):
	pass

RegymManager = None
RegymSummaryWriterPath = None

import ray
@ray.remote
class SharedVariable:
	def __init__(self, init_val=0):
		self.var = init_val
	def inc(self, n):
		self.var += n
	def get(self):
		return self.var
	def set(self, val):
		self.var = val

from . import environments
from . import util
from . import game_theory
from . import training_schemes
from . import rl_loops
from . import rl_algorithms
from . import logging_server
