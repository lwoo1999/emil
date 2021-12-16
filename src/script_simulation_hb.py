import numpy as np

from simulation import simulate, gen_by_covm

import pickle

f = open("../res/hb.bfsim", "rb")

dct = pickle.load(f)

hb = dct["data"]
hb_res = dct["res"]
hb_ans = dct["ans"]


f.close()

logcfs = hb_ans[:,2]
logcfs_mean = np.mean(logcfs)
logcfs_std = np.std(logcfs)

loglbols = hb_ans[:,0]
loglbols_mean = np.mean(loglbols)
loglbols_std = np.std(loglbols)

logbhs = hb["logbh_hb_vp06"]
logbhs_mean = np.mean(logbhs)
logbhs_std = np.std(logbhs)

logews = np.log10(hb["ew_broad_hb"])
logews_mean = np.mean(logews)
logews_std = np.std(logews)

corr = np.corrcoef([
    logews, logcfs, loglbols, logbhs
])

def single_simulate(i):
    data = hb[i]
    params_ = hb_res[i]
    loglbol_norm = (loglbols[i] - loglbols_mean) / loglbols_std
    logbh_norm = (logbhs[i] - logbhs_mean) / logbhs_std

    logew_norm, logcf_norm = gen_by_covm(corr, [loglbol_norm, logbh_norm])
    logew = logew_norm * logews_std + logews_mean
    logcf = logcf_norm * logcfs_std + logcfs_mean

    return [*simulate(data, params_, logcf), logew]

# if __name__ == "__main__":
#     from multiprocessing import Pool

#     p = Pool(60)
#     res = p.map(single_simulate, range(len(hb)))

#     np.savetxt("../res/hb.sim", np.array(res))
