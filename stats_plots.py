



# DATA EXPLORATION

close_counts = np.unique(np.sign(df_train[:,3]), return_counts=True)
plt.bar([str(i) for i in close_counts[0].astype(int)], close_counts[1]/np.sum(close_counts[1])*100)
plt.ylim(0, 100)
plt.show()

# MODEL METRICS

model_name = 'full_network_11epoch'
filepath = f'stats/{model_name}_stats.pickle'

d = pd.read_pickle(filepath)
print(list(d.keys()))

len(d['training_losses'])
len(d['train_loss'])
len(d['valid_loss'])
len(d['time_per_epoch'])

plt.plot(d['train_loss'], label = 'train loss')
plt.plot(d['valid_loss'], label = 'valid loss')
plt.legend(loc='upper right')
plt.title(f'{model_name} LOSS PER EPOCH')
# plt.savefig(f'{plots_dir}/{model_name}_LOSS_PLOT.png')
plt.show()


plt.plot(d['training_losses'], label = 'train loss')
plt.legend(loc='upper right')
plt.title(f'{model_name} LOSS PER TRAINING ITERATION')
# plt.savefig(f'{plots_dir}/{model_name}_LOSS_PLOT_ITER.png')
plt.show()

plt.plot(d['train_accuracy'], label = 'train accuracy')
plt.plot(d['valid_accuracy'], label = 'valid accuracy')
plt.legend(loc='upper right')
plt.title(f'{model_name} ACCURACY PER EPOCH')
# plt.savefig(f'{plots_dir}/{model_name}_LOSS_PLOT.png')
plt.show()