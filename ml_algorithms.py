import numpy as np
from utility import sigmoid


class LinearRegression:

    def __init__(self, num_features, lr=0.1):

        '''
        :param num_features: input dimension
        :param lr: learning rate
        '''

        self.n = num_features
        # self.theta = np.random.normal(size=(self.n, ))
        self.theta = np.zeros(shape=(2, ))
        self.learning_rate = lr


    def train(self, X, Y, epochs=1000, batch_size=32):

        '''

        :param X: examples input as Pandas dataframe
        :param Y: examples classes as Pandas dataframe
        :param epochs: number of epochs
        :param batch_size: batch size
        :return: history of mae during training process
        '''

        #X, Y = X.to_numpy(), Y.to_numpy()
        history = []

        for e in range(0, epochs):
            pred_error = 0
            mae = 0
            for i in range(0, X.shape[0]):
                predict = np.dot(X[i].T, self.theta)
                pred_error += (predict - Y[i]) * X[i]
                mae += abs(predict - Y[i])

                # Stochastic Gradient Descent
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
        #X, Y = X.to_numpy(), Y.to_numpy()

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


class LogisticRegression:

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
            accuracy = 0
            for i in range(0, X.shape[0]):
                x = np.dot(X[i].T, self.theta)
                predict = sigmoid(x)

                if predict >= 0.5:
                    cls = 1
                else:
                    cls = 0

                pred_error += (predict - Y[i]) * X[i]

                if cls == Y[i]:
                    accuracy += 1

                # Stochastic Gradient Descent
                if (i + 1) % batch_size == 0:
                    loss = self.learning_rate * pred_error / batch_size
                    self.theta -= loss

            accuracy /= X.shape[0]
            print('Epoch: {} | Accuracy: {} | Loss: {}'.format(e, accuracy, np.sum(pred_error)))
            history.append(accuracy)

        return history

    def test(self, X, Y):

        '''

        :param X: examples input as Pandas dataframe
        :param Y: examples class as Pandas dataframe
        :return:
        '''

        accuracy = 0
        X, Y = X.to_numpy(), Y.to_numpy()

        for i in range(0, X.shape[0]):
            x = np.dot(X[i].T, self.theta)
            predict = sigmoid(x)

            if predict >= 0.5:
                cls = 1
            else:
                cls = 0

            if cls == Y[i]:
                accuracy += 1

        print('Validation MAE: {}'.format(accuracy / X.shape[0]))

    def predict(self, X):

        '''
        :param X: examples input as Pandas dataframe
        :return: array of predictions
        '''

        X = X.to_numpy()
        y = []

        for i in range(0, X.shape[0]):
            x = np.dot(X[i].T, self.theta)
            predict = sigmoid(x)

            if predict >= 0.5:
                y.append(1)
            else:
                y.append(0)

        return np.asarray(y)


class NeuralNetwork:

    def __init__(self, attr_num, num_clss, hidden_layers):
        self.attr_num = attr_num
        self.num_clss = num_clss
        self.hidden_layers = hidden_layers

        self.syn = []
        # Normal distribution initialization with sigma=0.4 and mean=0
        self.syn.append(np.random.normal(0, 0.4, (attr_num, self.hidden_layers[0])))

        for i in range(0, len(self.hidden_layers)-1):
            self.syn.append(np.random.normal(0, 0.4, (self.hidden_layers[i], self.hidden_layers[i+1])))

        self.syn.append(np.random.normal(0, 0.4, (self.hidden_layers[len(self.hidden_layers)-1], num_clss)))


    def train(self, num_epochs, lr, X, Y, reg_l2=0, keep_prob=1.0, batch_size=64):
        """

        :param num_epochs: number of training epochs
        :param lr: learning rate
        :param X: training instance as python array with shape (number_of_examples, number_of_attributes)
        :param Y: training labels as python array with shape (number_of_examples, number_of_classes)
        :param hidden_layers: a list containing the number of units for each layer
        :param reg_l2: L2-regularization parameters
        :param keep_prob: dropout keep probability for activation node
        :return: training history as list of tuples with format (epoch, accuracy, loss)
        """
        k = 0

        #X = X.to_numpy()
        #y = y.to_numpy()

        history = []
        m = X.shape[0]

        while k < num_epochs:

            x_batches = np.split(X, X.shape[0] / batch_size)
            y_batches = np.split(Y, Y.shape[0] / batch_size)

            for x, y in zip(x_batches, y_batches):

                l = []
                all_l_deltas = []

                l.append(x.copy())
                l_tmp = x.copy()

                # forward pass

                for i in range(0, len(self.hidden_layers)):
                    l_tmp = np.tanh(np.dot(l_tmp, self.syn[i]))
                    l.append(l_tmp)

                #dropout
                '''if keep_prob != 1:
                    drop = np.random.rand(*l1.shape)
                    drop_l1 = (drop < keep_prob).astype('int')
                    l1 = l1 * drop_l1
                    l1 = l1 / keep_prob'''

                l.append(sigmoid(np.dot(l_tmp, self.syn[len(self.syn)-1])))

                # backprop
                l_error = l[len(l)-1] - y
                loss = np.sum(np.abs(l_error))
                l_delta = (1 / m) * np.dot(l[len(l)-2].T, l_error) + (reg_l2 / m) * self.syn[len(self.syn)-1]
                all_l_deltas.append(l_delta)
                assert l_delta.shape == self.syn[len(self.syn)-1].shape

                for i in range(0, len(self.hidden_layers), 1):
                    l_error = np.dot(l_error, self.syn[len(self.syn)-1-i].T) * (1 - np.power(l[len(l)-2-i], 2))

                    '''if keep_prob != 1:
                        l1_error_drop = l1_error * drop
                        l1_error_drop = l1_error_drop / keep_prob
                        l1_error = np.multiply(l1_error_drop, np.int64(l1 > 0))'''

                    l_delta = (1 / m) * np.dot(l[len(l)-3-i].T, l_error) + (reg_l2 / m) * self.syn[len(self.syn)-2-i]
                    all_l_deltas.append(l_delta)

                    assert l_delta.shape == self.syn[len(self.syn)-2-i].shape

                all_l_deltas = reversed(all_l_deltas)

                j = 0

                for x in all_l_deltas:
                    self.syn[j] -= lr * x
                    j += 1

            k += 1

            _, accuracy = self.predict(X, Y)

            history.append((k, accuracy, loss))

            print('Epoch: {}/{} | Accuracy: {:.2f} | Loss: {:.4f}'.format(k, num_epochs, accuracy, loss))

        return history

    def predict(self, X, y=None):
        accuracy = 0
        # forward pass
        l_tmp = X

        for i in range(0, len(self.hidden_layers)):
            l_tmp = np.tanh(np.dot(l_tmp, self.syn[i]))

        l_out = sigmoid(np.dot(l_tmp, self.syn[len(self.syn) - 1]))

        if self.num_clss == 1:
            preds = (l_out > 0.5) * 1
        else:
            preds = np.argmax(l_out, axis=1)
            if y is not None:
                y = np.argmax(y, axis=1)

        accuracy = None
        if y is not None:
            accuracy = np.sum(np.equal(preds, y) * 1)
            accuracy = accuracy / X.shape[0]

        return preds, accuracy


class KMeans:
    def __init__(self, num_clusters, data_points):
        """

        :param num_clusters: set the number of desired clusters
        :param data_points: data to be clustered
        """
        self.k = num_clusters
        self.data = data_points.to_numpy()

        self.__random_init__()

        self.assigned_clusters = np.zeros(self.data.shape[0])


    def __random_init__(self):
        """
        Choose initial centroids from data
        :return:
        """

        indexes = np.random.choice(np.arange(0, self.data.shape[0]), size=self.k, replace=False)
        self.centroids = self.data[indexes]
        print(self.centroids)

    def __assign_clusters__(self):
        """
        Assign each data point to a cluster
        :return: distortion
        """

        dist = np.zeros((self.k, ))
        distortion = 0

        for index in range(0, self.data.shape[0]):
            for i in range(0, self.k):
                dist[i] = np.linalg.norm(self.data[index] - self.centroids[i])

            self.assigned_clusters[index] = np.argmin(dist)
            distortion += np.min(dist)

        return distortion

    def __compute_centroids__(self):

        """
        Compute new centroid as average of cluster data points
        :return:
        """

        for i in range(0, self.k):
            cluster = np.argwhere(self.assigned_clusters == i)
            cluster_points = self.data[cluster].squeeze()
            self.centroids[i] = np.mean(cluster_points, axis=0)

        print(self.centroids)


    def train(self, num_iter):
        for i in range(0, num_iter):
            J = self.__assign_clusters__()
            print('Iteration: {} | Distortion: {}'.format(i, J))
            self.__compute_centroids__()

        return self.assigned_clusters


class AnomalyDetection:

    def __init__(self, data, eps=0.01):
        """
        :param data: features as Pandas dataframe
        :param eps: probability threshold for anomaly detection
        """
        self.data = data.copy().to_numpy()
        self.eps = eps

    def fit(self):
        self.mean = np.mean(self.data, axis=0)
        self.sigma = np.var(self.data, axis=0)

    def predict(self, x):
        x = x.copy().to_numpy()
        p = np.prod(1 / (np.sqrt(2 * np.pi) * self.sigma) * np.exp(- (x - self.mean) ** 2 / (2 * self.sigma ** 2)),
                    axis=1)
        return p < self.eps, p





