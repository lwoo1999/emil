import numpy as np
from scipy import integrate
from model import prepare_data, dust_models, disk_powlaw, blackbody, ec
from parcor import *

def components(params_):
    *params, _, mod = params_
    dust_model = dust_models[int(mod)]

    disk_amp, dust_temp, dust_lbol, av, cold_dust = params
    disk_amp = 10**disk_amp
    dust_lbol = 10**dust_lbol
    cold_dust = 10**cold_dust

    def optic(x):
        pl = disk_powlaw(x, disk_amp)
        return pl
    
    def nir(x):
        bb = blackbody(x, dust_temp, dust_lbol)
        cd = cold_dust * dust_model(x)
        return bb + cd
    
    return optic, nir

def analysis(params_):
    optic, nir = components(params_)

    loglbol = np.log10(5 * optic(0.3))
    loglnir = np.log10(integrate.quad(lambda x: nir(x)/x, 1., 10)[0])
    logcf = loglnir - loglbol

    return [loglbol, loglnir, logcf]

def analysis_(params_):
    optic, nir = components(params_)

    loglbol = np.log10(5 * optic(0.3))
    loglnirs = np.log10(nir(np.linspace(1,10)))
    logcfs = loglnirs - loglbol

    return logcfs

def analysis_catalog(data, params_):
    _, nir = components(params_)

    loglbol = data["logl5100"] + np.log10(9.26)
    loglnirs = np.log10(nir(np.linspace(1,10)))
    logcfs = loglnirs - loglbol

    return logcfs


