import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np
import pickle
import sys
sys.path.append('./')
import ablation_trial

# Lin = 1
jobid = '28336141'

# Lin = 0, hidden_dim = 10
#jobid = '28336296'

filename = f'ablation_data/input_dim_{jobid}.pickle'

with open(filename, 'rb') as handle:
    results = pickle.load(handle)

runtimes = []

input_dims = [k for k in results.keys() if isinstance(k, int)]

for k in input_dims:
    runtime = np.array([r.runtime for r in results[k]])
    runtimes.append(runtime)

vol_outs = []
vol_rels = []
for k in input_dims:
    vol_input = np.full(len(results[k]), 2**k)
    vol_output = np.array([r.upper_bound - r.lower_bound for r in results[k]])
    vol_rel = vol_output / vol_input 
    vol_outs.append(vol_output)
    vol_rels.append(vol_rel)

plt.boxplot(runtimes,
            labels=input_dims)
plt.ylabel('Execution Time (s)')
plt.xlabel('Input Dimension')
ax = plt.gca()
ax.set_yscale('log', base=2)
ax.yaxis.set_major_formatter(ScalarFormatter())
plt.savefig('input_dim_time.png')
plt.close()




plt.boxplot(vol_outs,
            labels=input_dims)
plt.savefig('input_dim_outvol.png')
plt.close()
plt.boxplot(vol_rels,
            labels=input_dims)
plt.savefig('input_dim_relvol.png')
plt.close()
