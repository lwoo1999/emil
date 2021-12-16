import numpy as np
from scipy.interpolate import CubicSpline

from model import *
from data import *

oiii = for_oiii()
oiii_ = oiii[oiii["loglbol"] > 46]
oiii_res = np.loadtxt("../output/oiii.res")

cond_oiii = (oiii_res[:,3] > -0.5) & (oiii_res[:,-2] < 50)
oiii_ = oiii_[cond_oiii]
oiii_res = oiii_res[cond_oiii]

def fit_spine(data, params_):
    rsr, wavelength, lum, lum_unc = prepare_data(data)
    *params, residual, mod = params_
    dust_model = dust_models[int(mod)]
    bands = np.log10([get_band(dust_model, params)(rsr, wav) for rsr, wav in zip(rsr, wavelength)])

    return CubicSpline(wavelength, lum - bands)