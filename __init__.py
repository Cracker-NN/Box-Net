import sys

sys.path.append("./hardware/__init__.py")
sys.path.append("./network/__init__.py")
sys.path.append("./preprocessing/__init__.py")
sys.path.append("./reader/__init__.py")

from hardware import *
from network import *
from preprocessing import *
from reader import *

LEARNING_RATE = (1e-3, 3e-4)

__VERSION__ = 0.1
__NAME__ = __name__
