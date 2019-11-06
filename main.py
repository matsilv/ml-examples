from ml_algorithms import LinearRegression, LogisticRegression, NeuralNetwork, KMeans, AnomalyDetection
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import pandas


def load_planar_dataset():
    np.random.seed(1)
    m = 400  # number of examples
    N = int(m / 2)  # number of points per class
    D = 2  # dimensionality
    X = np.zeros((m, D))  # data matrix where each row is a single example
    Y = np.zeros((m, 1), dtype='uint8')  # labels vector (0 for red, 1 for blue)
    a = 4  # maximum ray of the flower

    for j in range(2):
        ix = range(N * j, N * (j + 1))
        t = np.linspace(j * 3.12, (j + 1) * 3.12, N) + np.random.randn(N) * 0.2  # theta
        r = a * np.sin(4 * t) + np.random.randn(N) * 0.2  # radius
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        Y[ix] = j

    return X, Y


def load_2D_dataset():
    data = scipy.io.loadmat('data/data.mat')
    train_X = data['X']
    train_Y = data['y']
    test_X = data['Xval']
    test_Y = data['yval']

    plt.scatter(train_X[:, 0], train_X[:, 1], c=np.reshape(train_Y, -1), s=40, cmap=plt.cm.Spectral);

    return train_X, train_Y, test_X, test_Y


def plot_decision_boundary(model, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    Z, _ = model.predict(X=np.c_[xx.ravel(), yy.ravel()], y=None)
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    y = y.reshape(-1)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)


#X_train, y_train, X_test, y_test = load_2D_dataset()

#X, Y = load_planar_dataset()

df = pandas.read_csv('data/iris.csv')
X = df[df.columns[:-1]]
Y = df[[df.columns[-1]]]
Y = pandas.get_dummies(Y)
X = X.to_numpy()
Y = Y.to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.9, test_size=0.1, random_state=42)

model = NeuralNetwork(attr_num=X_train.shape[1], num_hidden=10, num_clss=y_train.shape[1])
model.train(num_epochs=10000, lr=0.1, X=X_train, y=y_train, reg_l2=1.0, keep_prob=1.0)
_, acc = model.predict(X=X_test, y=y_test)
print('Test set accuracy: {:.2f}'.format(acc))
#plot_decision_boundary(model, X_test, y=y_test)
plt.show()

