import numpy as np


class LinearRegression:

    def __init__(self, num_features, lr=0.1):

        '''
        :param num_features: input dimension
        :param lr: learning rate
        '''

        self.n = num_features
        self.theta = np.random.normal(size=(self.n, ))
        self.learning_rate = lr


    def train(self, X, Y, epochs=1000, batch_size=32):

        '''

        :param X: examples input as Pandas dataframe
        :param Y: examples classes as Pandas dataframe
        :param epochs: number of epochs
        :param batch_size: batch size
        :return: history of mae during training process
        '''

        X, Y = X.to_numpy(), Y.to_numpy()
        history = []

        for e in range(0, epochs):
            pred_error = 0
            mae = 0
            for i in range(0, X.shape[0]):
                predict = np.dot(X[i].T, self.theta)
                pred_error += (predict - Y[i]) * X[i]
                mae += abs(predict - Y[i])

                if (i + 1) % batch_size == 0:
                    loss = self.learning_rate * pred_error / X.shape[0]
                    self.theta -= loss

            print('Epoch: {} | MAE: {}'.format(e, mae))
            history.append(mae)

        return history

    def test(self, X, Y):

        '''

        :param X: examples input as Pandas dataframe
        :param Y: examples class as Pandas dataframe
        :return:
        '''

        mae = 0
        X, Y = X.to_numpy(), Y.to_numpy()

        for i in range(0, X.shape[0]):
            predict = np.dot(X[i].T, self.theta)
            mae += abs(predict - Y[i])

        print('Validation MAE: {}'.format(mae))

    def predict(self, X):

        '''
        :param X: examples input as Pandas dataframe
        :return: array of predictions
        '''

        X = X.to_numpy()
        y = []

        for i in range(0, X.shape[0]):
            predict = np.dot(X[i].T, self.theta)
            y.append(predict)

        return np.asarray(y)