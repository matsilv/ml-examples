import numpy as np
import matplotlib.pyplot as plt

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

########################################################################################################################
