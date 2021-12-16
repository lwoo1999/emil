import numpy as np
from astropy.io import fits

file = fits.open("../data/dr7_bh_Nov19_2013.fits")
data = file[1].data
file.close()


def for_hb():
    ret = data
    ret = ret[ret["z_hw"] < 0.851]
    ret = ret[ret["ew_narrow_hb"] != 0]
    ret = ret[ret["ew_broad_hb"] != 0]
    ret = ret[
        np.apply_along_axis(all, 1, ret["WISE1234"] / ret["WISE1234_ERR"] > 3)
        ]
    return ret


def for_mgii():
    ret = data
    ret = ret[ret["z_hw"] < 2.215]
    ret = ret[ret["z_hw"] > 0.429]
    ret = ret[ret["ew_mgii"] != 0]
    ret = ret[
        np.apply_along_axis(all, 1, ret["WISE1234"] / ret["WISE1234_ERR"] > 3)
        ]
    return ret


def for_oiii(ir = None):
    uppper = 0.797
    if ir:
        uppper = max(0, min(22 / ir - 1, 0.797))

    ret = data
    print(f"initial: {len(ret)}")
    ret = ret[ret["z_hw"] < uppper]
    ret = ret[ret["ew_oiii_5007"] != 0]
    print(f"select redshift: {len(ret)}")
    ret = ret[
        np.apply_along_axis(all, 1, ret["WISE1234"] / ret["WISE1234_ERR"] > 3)
        ]
    print(f"select wise: {len(ret)}")
    return ret


def for_civ():
    ret = data
    ret = ret[ret["z_hw"] > 2.228]
    ret = ret[ret["z_hw"] < 4.165]
    ret = ret[ret["ew_civ"] != 0]
    ret = ret[
        np.apply_along_axis(all, 1, ret["WISE1234"] / ret["WISE1234_ERR"] > 3)
        ]
    return ret