import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np
import pickle
import sys
sys.path.append('./')
import ablation_trial

#jobid = '28336182'
#jobid = '28336244'

gpu_to_jobid = {
    #1: 28808049,
    #1: 29631156,
    1: 29632026,
    2: 29632029,
    4: 29632034,
    #layers_29631156_gpus1.pickle
    #2: 28808131,
    #4: 28808275,
    #8: 28808342
}


gpu_to_res = {}
for count, jobid in gpu_to_jobid.items():
    filename = f'ablation_data/layers_{jobid}_gpus{count}.pickle'
    with open(filename, 'rb') as handle:
        results = pickle.load(handle)
    n_layers = [k for k in results.keys() if isinstance(k, int)]

    bern_runtimes = []
    bern_vols = []
    bern_ibf_runtimes = []
    bern_ibf_vols = []
    for k in n_layers:
        bern_runtime = np.array([r.bern_runtime for r in results[k]])[1:]
        bern_ibf_runtime = np.array([r.bern_ibf_runtime for r in results[k]])[1:]
        print(bern_runtime, bern_ibf_runtime)

        bern_vol = np.array([r.bern_vol.cpu() for r in results[k]])
        bern_ibf_vol = np.array([r.bern_ibf_vol[0] for r in results[k]])

        bern_runtimes.append(bern_runtime)
        bern_vols.append(bern_vol)
        bern_ibf_runtimes.append(bern_ibf_runtime)
        bern_ibf_vols.append(bern_ibf_vol)

    gpu_to_res[count] = (bern_runtimes, bern_vols, bern_ibf_runtimes, bern_ibf_vols)

plt.style.use('seaborn-v0_8-talk')
plt.figure(figsize=(8,4))
res = gpu_to_res[1]
bp3 = plt.boxplot(res[0], labels=n_layers, widths=0.35, patch_artist=True, showfliers=False, boxprops=dict(facecolor="C2"))
legend = [(bp3['boxes'][0], 'BERN-NN')]
for idx, (count, res) in enumerate(gpu_to_res.items()):
    bp4 = plt.boxplot(res[2],
                      labels=n_layers,
                      widths=0.35,
                      patch_artist=True,
                      showfliers=False,
                      boxprops=dict(facecolor=f"C{idx+3}"))
    legend.append((bp4['boxes'][0], f'IBF {count} GPUs'))
    plt.boxplot(res[2], labels=n_layers)
legend = tuple(zip(*legend))
print(legend)
plt.legend(legend[0], legend[1])
plt.ylabel('Execution Time (s)')
plt.xlabel('Hidden Layers')
plt.tight_layout()
plt.savefig('layers_time.pdf')
plt.close()

plt.figure(figsize=(8,4))
res = gpu_to_res[1]
plt.plot(n_layers, [r.mean() for r in res[1]], c='b', label='BERN-NN')
plt.plot(n_layers, [r.mean() for r in res[3]], c='r', label='BERN-NN-IBF')
plt.yscale('log')
plt.ylabel('Volume')
plt.xlabel('Hidden Layers')
plt.legend()
plt.tight_layout()
plt.savefig('layers_vol.pdf')
plt.close()

"""
res = gpu_to_res[1]
bp3 = plt.boxplot(res[1], labels=n_layers, widths=0.35, patch_artist=True, showfliers=False, boxprops=dict(facecolor="C2"))
for idx, (count, res) in enumerate(gpu_to_res.items()):
    bp4 = plt.boxplot(res[3],
                      labels=n_layers,
                      widths=0.35,
                      patch_artist=True,
                      showfliers=False,
                      boxprops=dict(facecolor=f"C{idx+3}"))
    plt.boxplot(res[2], labels=n_layers)
plt.yscale('log')
plt.ylabel('Volume')
plt.xlabel('Hidden Layers')
plt.savefig('layers_vol.png')
plt.close()
"""

"""
plt.boxplot(runtimes,
            labels=n_layers)
plt.ylabel('Execution Time (s)')
plt.xlabel('Hidden Layers')
ax = plt.gca()
ax.set_yscale('log', base=2)
ax.yaxis.set_major_formatter(ScalarFormatter())
plt.savefig('layers_time.png')
plt.close()


#plt.boxplot(runtimes,
#            labels=n_layers)
#plt.savefig('layers_time.png')
#plt.close()

plt.boxplot(vol_outs,
            labels=n_layers)
plt.savefig('layers_outvol.png')
plt.close()
plt.boxplot(vol_rels,
            labels=n_layers)
plt.savefig('layers_relvol.png')
plt.close()
"""
