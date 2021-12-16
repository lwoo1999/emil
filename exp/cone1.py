import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

n = 10000


_a_std = 0.23511969832958668
_b_std = 0.1988515228185253

def gen_data_frac(angle_mean, angle_std, sed_frac):
    angle = angle_std * np.random.randn(n) + angle_mean
    angle = angle[(angle < 1) & (angle > np.random.rand(n))]

    l = len(angle)

    sed = np.sqrt(_a_std**2 * sed_frac) * np.random.randn(l)

    a = None
    b = None
    while True:
        a = np.random.randn(l)
        b = np.random.randn(l)
        corr = np.corrcoef(a, b)[0, 1]
        if np.abs(corr) < 1e-4:
            break
    
    return a, b, angle, sed



def target(data, s_a, s_b):
    a, b, angle, sed = data
    a = s_a * a + sed + np.log10(angle)
    b = s_b * b + sed + np.log10(1-angle)

    return (np.std(a) - _a_std) ** 2 + (np.std(b) - _b_std) ** 2



def gen_data0(angle_mean, angle_std):
    angle = angle_std * np.random.randn(n) + angle_mean
    angle = angle[(angle < 1) & (angle > np.random.rand(n))]

    l = len(angle)


    a = None
    b = None
    while True:
        a = np.random.randn(l)
        b = np.random.randn(l)
        corr = np.corrcoef(a, b)[0, 1]
        if np.abs(corr) < 1e-4:
            break
    
    return a, b, angle


def target0(data, s_a, s_b):
    a, b, angle = data
    a = s_a * a + s_b * b + np.log10(angle)
    b = s_b * b + np.log10(1-angle)

    return (np.std(a) - _a_std) ** 2 + (np.std(b) - _b_std) ** 2