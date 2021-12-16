import numpy as np

from simulation import simulate, gen_by_covm

import pickle

f = open("../res/oiii.bfsim", "rb")

dct = pickle.load(f)

oiii = dct["data"]
oiii_res = dct["res"]
oiii_ans = dct["ans"]


f.close()

logcfs = oiii_ans[:,2]
logcfs_mean = np.mean(logcfs)
logcfs_std = np.std(logcfs)

loglbols = oiii_ans[:,0]
loglbols_mean = np.mean(loglbols)
loglbols_std = np.std(loglbols)

logbhs = oiii["logbh_hb_vp06"]
logbhs_mean = np.mean(logbhs)
logbhs_std = np.std(logbhs)

logews = np.log10(oiii["ew_oiii_5007"])
logews_mean = np.mean(logews)
logews_std = np.std(logews)

corr = np.corrcoef([
    logews, logcfs, loglbols, logbhs
])

def norm_corr(corr):
    return np.array([
        [corr[i, j] / np.sqrt(corr[i, i] * corr[j, j]) for j in range(4)]
        for i in range(4)
    ])

def get_new_corr(corr, n):
    corr_t = norm_corr(np.linalg.inv(corr))
    corr_t[0, 1] = -n
    corr_t[1, 0] = -n
    return norm_corr(np.linalg.inv(corr_t))

def gen_rands(corr):
    n = len(oiii)
    res, target = None, np.inf
    for _ in range(50):
        res_ = np.array([
            gen_by_covm(corr, [(loglbols[i] - loglbols_mean) / loglbols_std, (logbhs[i] - logbhs_mean) / logbhs_std])
            for i in range(n)
            ])
        corr_ = np.corrcoef([
            res_[:, 0], res_[:, 1], loglbols, logbhs
        ])
        target_ = np.sum((corr - corr_) ** 2)
        if target_ < target:
            res = res_
            target = target_
    return res

def single_simulate(params):
    i, corr, rand = params
    params_ = oiii_res[i]
    data = oiii[i]

    logew_norm, logcf_norm = rand
    logew = logew_norm * logews_std + logews_mean
    logcf = logcf_norm * logcfs_std + logcfs_mean

    return [*simulate(data, params_, logcf), logew]

# if __name__ == "__main__":
#     from multiprocessing import Pool

#     p = Pool(60)

#     ns = np.arange(-0.2, 0., 0.025)

#     for n in ns:
#         new_corr = get_new_corr(corr, n)
#         rand = gen_rands(corr)
#         res = np.array(p.map(single_simulate, [ [i, rand[i]] for i in range(len(oiii))]))

#         np.savetxt(f"../res/oiii{n}.sim", res)
