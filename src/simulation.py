import numpy as np
from model import dust_models, prepare_data, blackbody, disk_powlaw, ec, get_band, fit_for_storage
from analysis import *

def gen_by_covm(corr, a):
    # corr: 4*4 matrix
    # a   : 2   vector
    s11 = corr[0:2, 0:2]
    s22 = corr[2:4, 2:4]
    s12 = corr[0:2, 2:4]
    s21 = corr[2:4, 0:2]

    s = s11 - np.matmul(np.matmul(s12, np.linalg.inv(s22)), s21)
    mu = np.matmul(np.matmul(s12, np.linalg.inv(s22)), a)
    
    return np.random.multivariate_normal(mu, s)

def simulate(data, params_, new_logcf):
    rsr, wavelength, lum, lum_unc = prepare_data(data)
    *params, _, mod = params_
    dust_model = dust_models[int(mod)]
    _, _, logcf = analysis(params_)

    factor = new_logcf - logcf

    band = get_band(dust_model, params)

    disk_amp, dust_temp, dust_lbol, av, cold_dust = params
    band_ = get_band(dust_model, [
        disk_amp,
        dust_temp,
        dust_lbol + factor,
        av,
        cold_dust + factor
    ])

    lum_ = np.log10(band_(rsr, wavelength)) + lum - np.log10(band(rsr, wavelength))

    return fit_for_storage((rsr, wavelength, lum_, lum_unc), prepare_data=lambda x: x)