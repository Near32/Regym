from multiprocessing.managers import SyncManager, NamespaceProxy
import types 

class CustomManager(SyncManager):
	pass

def BuildProxy(target):
    dic = {'types': types}
    exec('''def __getattr__(self, key):
        result = self._callmethod('__getattribute__', (key,))
        if isinstance(result, types.MethodType):
            def wrapper(*args, **kwargs):
                self._callmethod(key, args)
            return wrapper
        return result''', dic)
    proxyName = target.__name__ + "Proxy"
    ProxyType = type(proxyName, (NamespaceProxy,), dic)
    ProxyType._exposed_ = tuple(dir(target))
    return ProxyType

RegymManager = None
RegymSummaryWriterPath = None

class SharedVariable:
	def __init__(self, init_val=0):
		self.var = init_val
	def inc(self, n):
		self.var += n
	def get(self):
		return self.var
	def set(self, val):
		self.var = val

import ray
RaySharedVariable = ray.remote(SharedVariable)

from . import environments
from . import util
from . import game_theory
from . import training_schemes
from . import rl_loops
from . import rl_algorithms
from . import logging_server
from . import pubsub_manager
