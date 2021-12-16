import numpy as np

from simulation import simulate, gen_by_covm

import pickle

f = open("../res/mgii.bfsim", "rb")

dct = pickle.load(f)

mgii = dct["data"]
mgii_res = dct["res"]
mgii_ans = dct["ans"]


f.close()

logcfs = mgii_ans[:,2]
logcfs_mean = np.mean(logcfs)
logcfs_std = np.std(logcfs)

loglbols = mgii_ans[:,0]
loglbols_mean = np.mean(loglbols)
loglbols_std = np.std(loglbols)

logbhs = mgii["logbh_mgii_vo09"]
logbhs_mean = np.mean(logbhs)
logbhs_std = np.std(logbhs)

logews = np.log10(mgii["ew_mgii"])
logews_mean = np.mean(logews)
logews_std = np.std(logews)

corr = np.corrcoef([
    logews, logcfs, loglbols, logbhs
])

def sim_test(i):
    loglbol_norm = (loglbols[i] - loglbols_mean) / loglbols_std
    logbh_norm = (logbhs[i] - logbhs_mean) / logbhs_std

    logew_norm, logcf_norm = gen_by_covm(corr, [loglbol_norm, logbh_norm])
    logew = logew_norm * logews_std + logews_mean
    logcf = logcf_norm * logcfs_std + logcfs_mean
    return [logew, logcf]

def single_simulate(i):
    data = mgii[i]
    params_ = mgii_res[i]
    loglbol_norm = (loglbols[i] - loglbols_mean) / loglbols_std
    logbh_norm = (logbhs[i] - logbhs_mean) / logbhs_std

    logew_norm, logcf_norm = gen_by_covm(corr, [loglbol_norm, logbh_norm])
    logew = logew_norm * logews_std + logews_mean
    logcf = logcf_norm * logcfs_std + logcfs_mean

    return [*simulate(data, params_, logcf), logew]

if __name__ == "__main__":
    from multiprocessing import Pool

    p = Pool(60)
    res = p.map(single_simulate, range(len(mgii)))

    np.savetxt("../res/mgii.sim", np.array(res))
