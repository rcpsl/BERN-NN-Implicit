import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np
import pickle
import sys
sys.path.append('./')
import ablation_trial

jobid = '28340634'
filename = f'ablation_data/hidden_dim_{jobid}.pickle'

with open(filename, 'rb') as handle:
    results = pickle.load(handle)


hidden_sizes = [k for k in results.keys() if isinstance(k, int)]

runtimes = []
for k in hidden_sizes:
    runtime = np.array([r.runtime for r in results[k]])
    runtimes.append(runtime)

vol_outs = []
vol_rels = []
for k in hidden_sizes:
    n_vars = results['n_vars']
    vol_input = np.full(len(results[k]), 2**n_vars)
    vol_output = np.array([r.upper_bound - r.lower_bound for r in results[k]])
    vol_rel = vol_output / vol_input 
    vol_outs.append(vol_output)
    vol_rels.append(vol_rel)

plt.boxplot(runtimes,
            labels=hidden_sizes)
plt.ylabel('Execution Time (s)')
plt.xlabel('Hidden Layer Dimension')
ax = plt.gca()
ax.set_yscale('log', base=2)
ax.yaxis.set_major_formatter(ScalarFormatter())

#plt.yscale('log', base=2)
#plt.ticklabel_format(axis='y', style='plain')
#yticks = np.geomspace(0.1, round(max([r.max() for r in runtimes]) + 0.5), num=5)
#plt.yticks(yticks, [f'{a:0.2f}' for a in yticks])
plt.minorticks_off()

plt.savefig('hidden_dim_time.png')
plt.close()
plt.boxplot(vol_outs,
            labels=hidden_sizes)
plt.savefig('hidden_dim_outvol.png')
plt.close()
plt.boxplot(vol_rels,
            labels=hidden_sizes)
plt.savefig('hidden_dim_relvol.png')
plt.close()
