

import pandas as pd
import matplotlib.pyplot as plt


model_name = 'full_network_4epoch'
filepath = '/stats/{model_name}_stats.pickle'

object = pd.read_pickle('stats/full_network_4epoch_stats.pickle')

len(object['training_losses'])

object['train_loss']
object['valid_loss']
object['time_per_epoch']

plt.plot(object['train_loss'], label = "train")
plt.plot(object['valid_loss'], label = "valid")
plt.legend(loc="upper right")
plt.show()

