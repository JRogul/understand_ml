import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from utils import metrics

def plot_decision_regions(X, y, classifier, resolution=0.02, return_metrics=False):
    """
    Plots decision regions of a classifier.

    Parameters
    ----------
    X : array-like, shape = [n_samples, n_features]
        Feature Matrix.
    y : array-like, shape = [n_samples]
        True class labels.
    classifier : Classifier object
        Must have a .predict method.
    resolution : float
        Granularity of the grid for plotting.

    Returns
    -------
    None

    """
    # setup marker generator and color map
    markers = ( 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))

    Z = np.array(classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T))
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], 
                    y=X[y == cl, 1],
                    alpha=0.6, 
                    color=cmap(idx),
                    edgecolor='black',
                    marker=markers[idx], 
                    label=cl)
    plt.title('Decision regions')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend(loc='upper left')
    plt.show()
    
    if return_metrics == True:
        preds, y_true = classifier.predict(X, y)
        print(f'Accuracy: {np.around(metrics.accuracy(y_true, preds), 2)*100}%' 
              f'\nRecall: {np.around(metrics.recall(y_true, preds), 2)}'
              f'\nPrecision: {np.around(metrics.precision(y_true, preds), 2)}'
              f'\nf1score:{np.around(metrics.f1_score(y_true, preds), 2)}')