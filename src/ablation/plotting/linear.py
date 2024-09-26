import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np
import pickle
import sys
sys.path.append('./')
import ablation_trial

jobid = '28679904'
filename = f'ablation_data/linear_{jobid}.pickle'

with open(filename, 'rb') as handle:
    results = pickle.load(handle)

lin_itr_numb = [k for k in results.keys() if isinstance(k, int)]

runtimes = []
for k in lin_itr_numb:
    runtime = np.array([r.runtime for r in results[k]])
    runtimes.append(runtime)

vol_outs = []
for k in lin_itr_numb:
    n_vars = results['n_vars']
    vol_outs.append(np.array([r.output_vol[0] for r in results[k]]))
    print(vol_outs)

plt.boxplot(runtimes,
            labels=lin_itr_numb)
plt.ylabel('Execution time (s)')
plt.xlabel('Linearization skip')

#plt.yscale('log', base=2)
#plt.ticklabel_format(axis='y', style='plain')
#yticks = np.geomspace(0.1, round(max([r.max() for r in runtimes]) + 0.5), num=5)
#plt.yticks(yticks, [f'{a:0.2f}' for a in yticks])
#plt.minorticks_off()

plt.savefig('linear_time.pdf')
plt.close()

plt.boxplot(vol_outs,
            labels=lin_itr_numb)
plt.ylabel('Output bound volume')
plt.xlabel('Linearization skip')
plt.savefig('linear_vol.pdf')
plt.close()
