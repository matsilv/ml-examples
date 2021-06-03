from ml_algorithms import LinearRegression, LogisticRegression, NeuralNetwork, KMeans, AnomalyDetection
from utility import scatter_plot_predictions
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import pandas
import os

########################################################################################################################



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


################################################ LINEAR REGRESSION #####################################################


'''# Read the dataset
df = pandas.read_csv(os.path.join('data', 'ex1data1.csv'))

# Get inputs
X = np.expand_dims(df['Population'].values, axis=1)
# Get targets
Y = np.expand_dims(df['Profit'].values, axis=1)

# Split training and test sets and standardize inputs and target
X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.9, test_size=0.1)

input_scaler = StandardScaler()
X_train = input_scaler.fit_transform(X_train)
X_test = input_scaler.transform(X_test)

target_scaler = StandardScaler()
y_train = target_scaler.fit_transform(y_train)
y_test = target_scaler.transform(y_test)

# Create and fit a linear regression model
linear_regression = LinearRegression(X_train.shape[1])
linear_regression.fit(X_train, y_train, learning_rate=0.1, batch_size=32)

# Plot predictions on training and test sets
scatter_plot_predictions(model=linear_regression, inputs=X_train, target=y_train)
scatter_plot_predictions(model=linear_regression, inputs=X_test, target=y_test)'''

################################################ LOGISTIC REGRESSION ###################################################


'''# Read the dataset
df = pandas.read_csv('data/iris.csv')
df = df[df['Class'] != 'Iris-setosa']

# Get inputs and target
X = df[df.columns[:-1]]
Y = df[[df.columns[-1]]]
Y = pandas.get_dummies(Y)

X = X.values
Y = np.expand_dims(Y.values[:, 0], axis=1)

# Split between training and test and standardize inputs
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25)

input_scaler = StandardScaler()
X_train = input_scaler.fit_transform(X_train)
X_test = input_scaler.transform(X_test)

# Create and fit a logistic regression model for binary classification
logistic_regression = LogisticRegression(X_train.shape[1])
logistic_regression.fit(X_train, y_train, learning_rate=0.1, batch_size=32)

# Compute accuracy on test set
preds = logistic_regression.predict(X_test)
print('Accuracy on test set: {:.2f}'.format(accuracy_score(preds, y_test)))'''

################################################ NEURAL NETWORK ##################################################################


df = pandas.read_csv('data/iris.csv')

# Get inputs and target
X = df[df.columns[:-1]]
Y = df[[df.columns[-1]]]
Y = pandas.get_dummies(Y)

X = X.values
Y = Y.values

# Split between training and test and standardize inputs
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25)

input_scaler = StandardScaler()
X_train = input_scaler.fit_transform(X_train)
X_test = input_scaler.transform(X_test)

model = NeuralNetwork(attr_num=X_train.shape[1], hidden_layers=[5, 5, 5], num_clss=y_train.shape[1])
model.fit(num_epochs=100,
          learning_rate=0.01,
          inputs=X_train,
          target=y_train,
          reg_l2=0.0,
          batch_size=64,
          optimizer='adam',
          beta_1=0.9,
          beta_2=0.999)
preds = model.predict(X_test, logits=False)
accuracy = accuracy_score(preds, y_test)

print('Test set accuracy: {:.2f}'.format(accuracy))
#plot_decision_boundary(model, X_test, y=y_test)
#plt.show()

