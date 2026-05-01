"""
# MatlPY
it is recommended to read the README.md for full documentation, the following text is a mere summary!
MatlyPY is a combination of Matplotlib.pyplot and Numpy with included ML presets. You can also do matrix & tensor
calculations. Although the lib is very fast its still a good lib.
"""
from .plotting import plot
from .mathematics import math
from .model import model
from .tools import tools  
from .cmath import cmath

model.tools = tools
math.cmath = cmath
