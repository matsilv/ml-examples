import numpy as np
from scipy.special import softmax
from sklearn.metrics import accuracy_score
import math

########################################################################################################################


class LinearRegression:

    def __init__(self, num_features):

        """
        :param num_features: int; input dimension.
        """

        self.dim = num_features
        self.theta = np.random.normal(size=(self.dim, 1))

    def fit(self,
            inputs: np.array,
            target: np.array,
            epochs: int = 1000,
            batch_size: int = 32,
            learning_rate: float = 0.1):

        """
        Train the linear regression model.
        :param inputs: numpy.array; input samples.
        :param target: numpy.array; target samples.
        :param epochs: int; number of epochs.
        :param batch_size: int; batch size.
        :param learning_rate: float; learning rate.
        :return: list; history of mae during training process.
        """

        history = []

        for e in range(0, epochs):

            batch_idxes = np.random.choice(np.arange(len(inputs)), batch_size, replace=False)
            batch_inputs = inputs[batch_idxes]
            batch_targets = target[batch_idxes]
            predictions = self.predict(batch_inputs)
            pred_error = np.sum((predictions - batch_targets) * batch_inputs, axis=0)

            loss = np.expand_dims(learning_rate * pred_error / batch_size, axis=1)
            self.theta -= loss

            if e % 100 == 0:
                val_preds = self.predict(inputs)
                mae = np.mean(abs(val_preds - target))
                print('Epoch: {} | MAE: {}'.format(e, mae))

            history.append(mae)

        return history

    def predict(self, inputs: np.array) -> np.array:
        """
        Make predictions given inputs.
        :param inputs: numpy.array; input samples.
        :return: predictions: numpy.array; predictions.
        """

        return np.dot(inputs, self.theta)

########################################################################################################################


class LogisticRegression:

    def __init__(self, num_features):

        """
        :param num_features: int; input dimension.
        """

        self.dim = num_features
        self.theta = np.random.normal(size=(self.dim, 1))

    def fit(self,
            inputs: np.array,
            target: np.array,
            epochs: int = 1000,
            batch_size: int = 32,
            learning_rate: float = 0.1):

        """
        Train the linear regression model.
        :param inputs: numpy.array; input samples.
        :param target: numpy.array; target samples.
        :param epochs: int; number of epochs.
        :param batch_size: int; batch size.
        :param learning_rate: float; learning rate.
        :return: list; history of mae during training process.
        """

        history = []

        for e in range(0, epochs):

            batch_idxes = np.random.choice(np.arange(len(inputs)), batch_size, replace=False)
            batch_inputs = inputs[batch_idxes]
            batch_targets = target[batch_idxes]
            predictions = self.predict(batch_inputs, logits=True)
            pred_error = np.sum((predictions - batch_targets) * batch_inputs, axis=0)

            loss = np.expand_dims(learning_rate * pred_error / batch_size, axis=1)
            self.theta -= loss

            if e % 100 == 0:
                val_preds = self.predict(inputs)
                accuracy = accuracy_score(val_preds, target)
                print('Epoch: {} | Accuracy: {}'.format(e, accuracy))

            history.append(accuracy)

        return history

    def predict(self, inputs: np.array, logits: bool = False, threshold: float = 0.5) -> np.array:
        """
        Make predictions given inputs.
        :param inputs: numpy.array; input samples.
        :param threshold: float; threshold for binary classification.
        :return: predictions: numpy.array; predictions.
        """

        preds = softmax(np.dot(inputs, self.theta), axis=1)
        if not logits:
            return preds >= threshold
        else:
            return preds

########################################################################################################################


class NeuralNetwork:

    def __init__(self, attr_num, num_clss, hidden_layers):
        self.attr_num = attr_num
        self.num_clss = num_clss
        self.hidden_layers = hidden_layers

        # Neural network weights
        self.syn = []

        # Normal distribution initialization with sigma=0.4 and mean=0
        self.syn.append(np.random.normal(0, 0.4, (attr_num, self.hidden_layers[0])))

        for i in range(0, len(self.hidden_layers)-1):
            self.syn.append(np.random.normal(0, 0.4, (self.hidden_layers[i], self.hidden_layers[i+1])))

        self.syn.append(np.random.normal(0, 0.4, (self.hidden_layers[len(self.hidden_layers)-1], num_clss)))

    def fit(self,
            inputs: np.array,
            target: np.array,
            num_epochs: int = 100,
            learning_rate: float = 0.1,
            reg_l2: float = 0.0,
            batch_size: int = 32,
            optimizer: str = 'sgd',
            beta_1: float = 0.9,
            beta_2: float = 0.999,
            epsilon: float = 1e-7):

        """
        Fit the neural network model.
        :param inputs: numpy.array; input samples.
        :param target: numpy.array; target samples.
        :param num_epochs: int; number of epochs.
        :param learning_rate: float; learning rate.
        :param reg_l2: float; L2-regularization paramter.
        :param batch_size: int; batch size.
        :param optimizer: string; weights optimizer.
        :param beta_1: float; exponential decay for 1st moment estimates.
        :param beta_2: float; exponential decay for 2nd moment estimates.
        :param epsilon: float; small constant for numerical stability.
        :return:
        """

        assert optimizer in ['sgd', 'rmsprop', 'adam'], "Illegal optimizer"

        history = []
        num_samples = inputs.shape[0]

        # Exponentially moving averages initialization
        all_v_deltas = []
        all_s_deltas = []
        all_s_deltas_sign = []

        for s in self.syn:
            all_v_deltas.append(np.zeros_like(s))
            all_s_deltas.append(np.zeros_like(s))
            all_s_deltas_sign.append((np.ones_like(s)))

        for epoch in range(1, num_epochs):

            x_batches = np.split(inputs, num_samples / batch_size)
            y_batches = np.split(target, num_samples / batch_size)

            for x_batch, y_batch in zip(x_batches, y_batches):

                layer_output = []
                all_layer_deltas = []

                layer_output.append(x_batch.copy())
                layer_temp_output = x_batch.copy()

                # Forward pass
                for i in range(0, len(self.hidden_layers)):
                    layer_temp_output = np.tanh(np.dot(layer_temp_output, self.syn[i]))
                    layer_output.append(layer_temp_output)

                layer_output.append(softmax(np.dot(layer_temp_output, self.syn[len(self.syn)-1]),
                                            axis=1))

                # Backprop
                layer_error = layer_output[len(layer_output)-1] - y_batch
                loss = np.sum(np.abs(layer_error))
                layer_delta = (1 / num_samples) * np.dot(layer_output[len(layer_output)-2].T, layer_error) + \
                              (reg_l2 / num_samples) * self.syn[len(self.syn)-1]

                layer_delta_v = beta_1 * all_v_deltas[len(self.syn)-1] + (1 - beta_1) * layer_delta
                all_v_deltas[len(self.syn)-1] = layer_delta_v
                layer_delta_s = beta_2 * all_s_deltas[len(self.syn)-1] + (1 - beta_2) * layer_delta ** 2
                all_s_deltas[len(self.syn)-1] = layer_delta_s

                assert layer_delta.shape == self.syn[len(self.syn)-1].shape

                for i in range(0, len(self.hidden_layers)):
                    # Derivative of tanh function
                    layer_error = np.dot(layer_error, self.syn[len(self.syn)-1-i].T) * \
                                  (1 - np.power(layer_output[len(layer_output)-2-i], 2))

                    layer_delta = (1 / num_samples) * np.dot(layer_output[len(layer_output)-3-i].T, layer_error) + \
                                  (reg_l2 / num_samples) * self.syn[len(self.syn)-2-i]

                    layer_delta_v = beta_1 * all_v_deltas[len(self.syn)-2-i] + (1 - beta_1) * layer_delta
                    all_layer_deltas.append(layer_delta)
                    all_v_deltas[len(self.syn)-2-i] = layer_delta_v

                    l_delta_s = beta_2 * all_s_deltas[len(self.syn)-2-i] + (1 - beta_2) * layer_delta ** 2
                    all_layer_deltas.append(layer_delta)
                    all_s_deltas[len(self.syn)-2-i] = l_delta_s

                    assert layer_delta.shape == self.syn[len(self.syn)-2-i].shape

                if optimizer == 'sgd':
                    for idx, delta in enumerate(all_v_deltas):
                        self.syn[idx] -= learning_rate * delta
                elif optimizer == 'adam':
                    for idx, (v_delta, s_delta) in enumerate(zip(all_v_deltas, all_s_deltas)):
                        self.syn[idx] -= learning_rate * v_delta / (np.sqrt(s_delta) + epsilon)

            # Compute accuracy
            preds = self.predict(inputs, logits=False)
            accuracy = accuracy_score(preds, target)

            print(f'Epoch: {epoch}/{num_epochs} | Loss: {loss} | Accuracy: {accuracy}')

        return history

    def predict(self, inputs, logits=True, threshold=0.5):
        output = inputs

        for i in range(0, len(self.hidden_layers)):
            output = np.tanh(np.dot(output, self.syn[i]))

        output = np.dot(output, self.syn[len(self.syn) - 1])

        if not logits:
            output = softmax(output, axis=1)
            output = output >= threshold

            return output


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





