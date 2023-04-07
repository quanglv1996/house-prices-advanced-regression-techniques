import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.append('../')
path_scripts = './../scripts'
path_chart = './../chart'


# Draw chart
def draw_chart(name_png, values, labels, name_chart, label_x, label_y, figsize=(20, 14), fontsize=50, rotation=30):
    index = np.arange(len(labels))
    plt.figure(1)
    plt.figure(figsize=figsize)
    plt.bar(index, values)
    plt.xlabel(label_x, fontsize=fontsize)
    plt.ylabel(label_y, fontsize=fontsize)
    plt.xticks(index, labels, fontsize=int(fontsize), rotation=rotation)
    plt.yticks(fontsize=int(fontsize), rotation=0)
    plt.title(name_chart, fontsize=int(fontsize * 1.3))
    plt.savefig(os.path.join(path_chart, name_png))


def draw_model_history(n_folds, histories, path_save):
    colors = ['black', 'red', 'green', 'blue', 'gray', 'yellow', 'violet', 'orange', 'lime', 'olive']
    plt.figure(figsize=(16, 9))
    plt.title('Train AUC-ROC vs Val AUC-ROC')
    for i in range(n_folds):
        plt.plot(histories[i].history['auroc'], label='Train AUC-ROC Fold ' + str(i + 1), color=colors[i])
        plt.plot(histories[i].history['val_auroc'], label='Val AUC-ROC Fold ' + str(i + 1), color=colors[i],
                 linestyle="dashdot")
    plt.legend()
    plt.savefig(os.path.join(path_save, 'train_val_auc.png'))