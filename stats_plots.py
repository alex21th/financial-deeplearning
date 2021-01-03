
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# MODEL METRICS

model_name = 'best_model'
# model_name = 'PLOT_edit_lr0.0001_TH_c100_t10_b64_h32_e3'
filepath = f'stats/{model_name}_stats.pickle'

measures = pd.read_pickle(filepath)
print(list(measures.keys()))

len(measures['training_losses'])
len(measures['train_loss'])
len(measures['valid_loss'])
len(measures['time_per_epoch'])

plt.plot(measures['train_loss'], label = 'train loss')
plt.plot(measures['valid_loss'], label = 'valid loss')
plt.legend(loc='upper right')
plt.title(f'{model_name} LOSS PER EPOCH')
# plt.savefig(f'{plots_dir}/{model_name}_LOSS_PLOT.png')
plt.show()


plt.plot(measures['training_losses'], label = 'train loss')
plt.legend(loc='upper right')
plt.title(f'{model_name} LOSS PER TRAINING ITERATION')
# plt.savefig(f'{plots_dir}/{model_name}_LOSS_PLOT_ITER.png')
plt.show()

plt.plot(measures['train_accuracy'], label = 'train accuracy')
plt.plot(measures['valid_accuracy'], label = 'valid accuracy')
plt.plot(measures['train_th_accuracy'], label = 'train th accuracy')
plt.plot(measures['valid_th_accuracy'], label = 'valid th accuracy')
plt.legend(loc='upper right')
plt.title(f'{model_name} ACCURACY PER EPOCH')
# plt.savefig(f'{plots_dir}/{model_name}_LOSS_PLOT.png')
plt.show()