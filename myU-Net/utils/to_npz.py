import joblib
import numpy as np

path = 'adults'
i = 0
for fold in range(5):
    x_train, y_train, train_dataset_desc = joblib.load('train_adults_aug_' + str(fold) + '.pkl')
    x_train_patches = np.rollaxis(x_train, 1, 5)
    y_train_patches = np.rollaxis(y_train, 1, 5)
    for j in range(x_train_patches.shape[0]):
        x = x_train_patches[j]
        y = y_train_patches[j]
        np.savez(path + '/' + str(i) + '.npz', x = x, y = y)
        i += 1