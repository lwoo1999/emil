import numpy as np
from astropy.cosmology import Planck15

from rsr import RSR

wavelength = np.array([
    1.235,
    1.662,
    2.159
])  # um

zeropoint = np.array([
    3.129E-13,
    1.133E-13,
    4.283E-14
])  # W/cm^2/um

zeropoint_mjy = np.array([
    1.594e6,
    1.024e6,
    6.667e5
])

zeropoint_mjy_unc = np.array([
    27.8e3,
    20.0e3,
    12.6e3
]) / zeropoint_mjy

zeropoint_unc = np.array([
    5.464E-15,
    2.212E-15,
    8.053E-16
]) / zeropoint

def mag_to_flux(mag, mag_unc):
    flux = 10 ** (-mag/2.5 + np.log10(zeropoint_mjy))
    flux_unc = np.sqrt((mag_unc / 2.5) ** 2 + (zeropoint_mjy_unc / np.log(10)) ** 2) * np.log(10) * flux
    return flux, flux_unc

def mag_to_lum(mag, mag_unc, z):
    dist = Planck15.luminosity_distance(z).value * 3.08568e+24  # cm
    lum = -mag/2.5 + np.log10(zeropoint) + np.log10(wavelength) + np.log10(4*np.pi*dist**2) + 7 # J -> erg
    lum_unc = np.sqrt((mag_unc / 2.5) ** 2 + (zeropoint_unc / np.log(10)) ** 2)
    return lum, lum_unc

rsr = []

for file in ["J","H","K"]:
    data = np.loadtxt(f"../data/rsr/{file}")
    rsr_ = RSR(data[:,0], data[:,1])
    rsr_.normalise()
    rsr.append(rsr_)