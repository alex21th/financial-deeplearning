#!/usr/bin/env python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

epochs = 30
technical_indicators = (False,True)
seeds = (1,2,3,4,5)
hidden = (32,64,128)
batch = (256,512,1024)

STATS_DIR = 'new_stats/'

models = {}
i,j = 0, 0
for t in technical_indicators:
    for h in hidden:
        for b in batch:
            accuracy = []
            for s in seeds:
                model_name = f'{t}_s{s}_h{h}_b{b}_e{epochs}'
                filepath = f'{STATS_DIR}/{model_name}_stats.pickle'
                measures = pd.read_pickle(filepath)
                accuracy.append(measures['valid_th_accuracy'])
                i += 1
            models[(t, h, b)] = np.mean(np.array(accuracy), axis=0)
            j += 1

print(f'       Total models (with seed): {i}')
print(f'Total models (averaged by seed): {j}')
print(f'  Dictionary of averaged models: {len(models.keys())}')

order_models = models.copy()
for k, v in order_models.items():
    order_models[k] = np.argsort(-v) + 1


max_models = models.copy()
for k, v in max_models.items():
    max_models[k] = np.max(max_models[k])

sorted_models = {}
sorted_keys = sorted(max_models, key=max_models.get, reverse=True)

for k in sorted_keys:
    sorted_models[k] = max_models[k]

# SEE AND GET RESULTS:

sorted_models
top_model = list(sorted_models.keys())[0]
models[top_model]
order_models[top_model]

# PLOT SOME RESULTS:

plt.plot(models[top_model], label = 'valid th accuracy')
plt.legend(loc='upper right')
plt.title(f'ACCURACY PER EPOCH')
# plt.savefig(f'{plots_dir}/{model_name}_LOSS_PLOT.png')
plt.show()