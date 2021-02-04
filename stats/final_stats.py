#!/usr/bin/env python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')

epochs = 30
seeds = (1,2,3,4,5)
batch = (1024,2048,4096,8192,16384)

STATS_DIR = 'final_stats/'  # !!!! CHANGE IF NECESSARY !!!!

def compute_results(metric):
    models = {}
    i,j = 0, 0
    for b in batch:
        accuracy = []
        for s in seeds:
            model_name = f's{s}_b{b}_e{epochs}'
            filepath = f'{STATS_DIR}{model_name}_stats.pickle'
            measures = pd.read_pickle(filepath)
            accuracy.append(measures[metric])
            i += 1
        models[b] = np.mean(np.array(accuracy), axis=0)
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

    return models, sorted_models, order_models


def plot_model(models, sorted_models, number):
    top_model = list(sorted_models.keys())[number-1]
    plt.plot(models[top_model], label = 'accuracy')
    plt.legend(loc='upper right')
    plt.title(f'ACCURACY PER EPOCH')
    plt.xticks(np.arange(0, len(models[top_model])))
    # plt.savefig(f'{plots_dir}/{model_name}_LOSS_PLOT.png')
    plt.show()


models, sorted_models, order_models = compute_results('valid_accuracy')
sorted_models
plot_model(models, sorted_models, 1)


th_models, th_sorted_models, th_order_models = compute_results('valid_th_accuracy')
th_sorted_models
plot_model(th_models, th_sorted_models, 1)