# for run in ipython

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

from model import *
from data import *
from analysis import *

oiii = for_oiii()
oiii = oiii[oiii["loglbol"] > 46]

oiii_res = np.loadtxt("../res/oiii.res")