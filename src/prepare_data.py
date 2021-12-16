import numpy as np
import jhk
import wise
import ugriz

def prepare_data(data):
    z = data["z_hw"]
    wavelength = []
    rsr = []
    lum = []
    lum_unc = []

    lum_ugriz, lum_ugriz_unc = ugriz.mag_to_lum(data["ugriz_dered"], data["ugriz_err"], z)
    for i in range(5):
        if data["ugriz_err"][i] != 0:
            wavelength.append(ugriz.wavelength[i]/(1+z))
            rsr.append(ugriz.rsr[i].redshift(z))
            lum.append(lum_ugriz[i])
            lum_unc.append(lum_ugriz_unc[i])

    lum_jhk, lum_jhk_unc = jhk.mag_to_lum(data["jhk"], data["jhk_err"], z)
    for i in range(3):
        if data["jhk_err"][i] != 0:
            wavelength.append(jhk.wavelength[i]/(1+z))
            rsr.append(jhk.rsr[i].redshift(z))
            lum.append(lum_jhk[i])
            lum_unc.append(lum_jhk_unc[i])

    lum_wise, lum_wise_unc = wise.mag_to_lum(data["wise1234"], data["wise1234_err"], z)
    for i in range(4):
        if data["wise1234_err"][i] != 0:
            wavelength.append(wise.wavelength[i]/(1+z))
            rsr.append(wise.rsr[i].redshift(z))
            lum.append(lum_wise[i])
            lum_unc.append(lum_wise_unc[i])

    return rsr, wavelength, lum, lum_unc


def prepare_data_for_cigale(data):
    res = []

    flux_ugriz, flux_ugriz_unc = ugriz.mag_to_flux(data["ugriz_dered"], data["ugriz_err"])
    for i in range(5):
        if data["ugriz_err"][i] != 0:
            res.extend([flux_ugriz[i], flux_ugriz_unc[i]])
        else:
            res.extend([-1., -1.])

    flux_jhk, flux_jhk_unc = jhk.mag_to_flux(data["jhk"], data["jhk_err"])
    for i in range(3):
        if data["jhk_err"][i] != 0:
            res.extend([flux_jhk[i], flux_jhk_unc[i]])
        else:
            res.extend([-1., -1.])
    
    flux_wise, flux_wise_unc = wise.mag_to_flux(data["wise1234"], data["wise1234_err"])
    for i in range(4):
        if data["wise1234_err"][i] != 0:
            res.extend([flux_wise[i], flux_wise_unc[i]])
        else:
            res.extend([-1., -1.])
            
    return f"{data['sdss_name']} {data['z_hw']} " + " ".join(map(str, res))