import glob
import matplotlib.pyplot as plt
import numpy as np
import pickle
import sys
sys.path.append('./')
import ablation_trial
import scipy.stats

def mean_confidence_interval(data, confidence=0.95):
    # Taken from stackoverflow:
    # https://stackoverflow.com/questions/15033511/compute-a-confidence-interval-from-sample-data
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h


# Lin=1, n_vars=5, hidden_dim=100, 8 A100
jobid = '28175579'

# Lin=0, n_vars=3, hidden_dim=50, 4 A100
#jobid = '28188317'
#jobid = '28188633'
#jobid = '28199085'

#jobid = '28199115'

result_filenames = sorted(glob.glob(f'results_strong/{jobid}*.pickle'))

results = {}

for filename in result_filenames:
    with open(filename, 'rb') as handle:
        results[filename] = pickle.load(handle)

runtimes = {}

for result in results.values():
    num_gpus = int(list(result.keys())[0])
    data = list(result.values())[0]
    runtimes[num_gpus] = np.array([datum.runtime for datum in data])

num_gpus = list(runtimes.keys())

ideal_par = np.mean(runtimes[1]) / np.array([int(k) for k in runtimes.keys()])
plt.plot(num_gpus, ideal_par, linestyle='dashed', color='black')

mean_runtimes, lo, hi = zip(*[mean_confidence_interval(runtime) for runtime in runtimes.values()])

# verify that the 95% confidence internval is within 5% of
# the sample mean for each GPU count.
# If it is, we can exclude them from the plot.
for i in range(len(mean_runtimes)):
    m, l, h = mean_runtimes[i], lo[i], hi[i]
#    assert m - 0.05 * m < l
#    assert m + 0.05 * m > h

plt.scatter(num_gpus, mean_runtimes, marker='x', color='red')

plt.yscale('log')
plt.xscale('log')
plt.xticks(num_gpus, num_gpus)
yticks = np.geomspace(np.min(ideal_par) - 0.5, np.max(ideal_par) + 0.5, num=5)
plt.yticks(yticks, [f'{a:0.2f}' for a in yticks])
plt.minorticks_off()

plt.grid()
plt.ylabel('Execution Time (s)')
plt.xlabel('Number of GPUs')
plt.savefig('strong_scaling.png')
