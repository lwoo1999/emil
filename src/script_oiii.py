from data import for_oiii
from model import fit_for_storage

from multiprocessing import Pool
import numpy as np

pool = Pool(40)

oiii = for_oiii()                  # 15756
oiii = oiii[oiii["loglbol"] > 46]  # 1927

res = pool.map(fit_for_storage, oiii)

np.savetxt("../output/oiii.res", np.array(res))