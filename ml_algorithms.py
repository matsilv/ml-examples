import numpy as np
import pandas


class LinearRegression:

    def __init__(self, num_features, lr=0.01):
        self.n = num_features
        self.theta = np.random.normal(size=(self.n, ))
        self.learning_rate = lr

    def train(self, X, Y, epochs=1000):
        X, Y = X.to_numpy(), Y.to_numpy()

        for e in range(0, epochs):
            pred_error = 0
            mae = 0
            for i in range(0, X.shape[0]):
                predict = np.dot(X[i].T, self.theta)
                pred_error += (predict - Y[i]) * X[i]
                mae += abs(predict - Y[i])

            loss = self.learning_rate * pred_error / X.shape[0]
            self.theta -= loss
            print('Epoch: {} | MAE: {}'.format(e, mae))