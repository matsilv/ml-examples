import numpy as np
import matplotlib.pyplot as plt

########################################################################################################################


def sigmoid(x, deriv=False):

    if(deriv == True):
        return  x * (1 - x)

    return 1 / (1 + np.exp(-x))

########################################################################################################################


def scatter_plot_predictions(model, inputs, target):
    sorted_idxes = np.squeeze(np.argsort(inputs, axis=0))
    preds = model.predict(inputs)
    inputs = inputs[sorted_idxes]
    target = target[sorted_idxes]
    preds = preds[sorted_idxes]
    plt.scatter(inputs, target)
    plt.plot(inputs, preds)
    plt.show()