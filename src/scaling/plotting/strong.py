import glob
import matplotlib.pyplot as plt
import numpy as np
import pickle
import sys
sys.path.append('./')
import ablation_trial

jobid = '28167426'

result_filenames = sorted(glob.glob(f'results_strong/{jobid}*.pickle'))

results = {}

for filename in result_filenames:
    with open(filename, 'rb') as handle:
        results[filename] = pickle.load(handle)

runtimes = {}

for result in results.values():
    num_gpus = list(result.keys())[0]
    data = list(result.values())[0]
    runtimes[num_gpus] = np.array([datum.runtime for datum in data])

plt.boxplot(runtimes.values(), labels=runtimes.keys())
plt.ylabel('Runtime (s)')
plt.xlabel('GPU Count')
plt.savefig('strong_scaling.png')

