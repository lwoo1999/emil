import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

n = 10000

def gen_data(angle_mean, angle_std):
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

def gen_data2(angle_mean, angle_std):
    angle = angle_std * np.random.randn(n) + angle_mean
    angle = angle[(angle < 1) & (angle > np.random.rand(n))]

    l = len(angle)

    a = None
    b1, b2 = None, None
    while True:
        a = np.random.randn(l)
        b1 = np.random.randn(l)
        b2 = np.random.randn(l)
        corr1 = np.corrcoef(a, b1)[0, 1]
        corr2 = np.corrcoef(a, b2)[0, 1]
        if np.abs(corr1) < 1e-4 and np.abs(corr2) < 1e-4:
            break
    
    return a, b1, b2, angle

_a_std = 0.23511969832958668
_b_std = 0.1988515228185253


# torus only
def target(data, s_a, s_b):
    a, b, angle = data
    a = s_a * a + np.log10(angle)
    b = s_b * b + np.log10(1-angle)

    return (np.std(a) - _a_std) ** 2 + (np.std(b) - _b_std) ** 2

def target2(data, s_a, s_b):
    a, b1, b2, angle = data
    a = s_a * a + np.log10(angle)
    b1 = s_b * b1 + np.log10(1-angle)
    b2 = s_b * b2 + np.log10(1-angle)

    return (np.std(a) - _a_std) ** 2 + (np.std(b1) - _b_std) ** 2 + (np.std(b2) - _b_std) ** 2

def simulate(angle_mean, angle_std):
    data = gen_data(angle_mean, angle_std)

    res = opt.minimize(
        lambda s: target(data, *s), 
        [_a_std, _b_std],
        bounds=opt.Bounds(
            [0, 0],
            [np.inf, np.inf]
        )
    )

    s_a, s_b = res.x
    a, b, angle = data
    a = s_a * a + np.log10(angle)
    b = s_b * b + np.log10(1-angle)

    return np.corrcoef(a, b)[0, 1]


# # with polar dust
# def target_p(data, polar_frac, s_a, s_b):
#     a, b, angle = data
#     a = s_a * a + np.log10(angle)
#     b_torus = s_b * b - np.log10(angle)
#     b_polar = s_b * b + np.log10(angle)
#     b = np.log10(10 ** (b_torus * (1 - polar_frac)) + 10 ** (b_polar * polar_frac))

#     return (np.std(a) - _a_std) ** 2 + (np.std(b) - _b_std) ** 2


# def simulate_p(angle_mean, angle_std, polar_frac):
#     data = gen_data(angle_mean, angle_std)

#     res = opt.minimize(
#         lambda s: target(data, polar_frac, *s), 
#         [_a_std, _b_std],
#         bounds=opt.Bounds(
#             [0, 0],
#             [np.inf, np.inf]
#         )
#     )

#     s_a, s_b = res.x
#     a, b, angle = data
#     a = s_a * a + np.log10(angle)
#     b_torus = s_b * b - np.log10(angle)
#     b_polar = s_b * b + np.log10(angle)
#     b = np.log10(10 ** (b_torus * (1 - polar_frac)) + 10 ** (b_polar * polar_frac))

#     return np.corrcoef(a, b)[0, 1]