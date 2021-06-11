from ml_algorithms import LinearRegression, LogisticRegression, NeuralNetwork, KMeans
from utility import scatter_plot_predictions
from sklearn.datasets._samples_generator import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import pandas
import os

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


'''df = pandas.read_csv('data/iris.csv')

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

model = NeuralNetwork(num_features=X_train.shape[1],
                      hidden_layers=[5, 5, 5],
                      num_clss=y_train.shape[1])

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
y_test = np.argmax(y_test, axis=1)
accuracy = accuracy_score(preds, y_test)

print('Test set accuracy: {:.2f}'.format(accuracy))'''

##################################################### K-MEANS ##########################################################


X, y_true = make_blobs(n_samples=100, centers=3, cluster_std=0.6)

plt.scatter(X[:, 0], X[:, 1])
plt.show()

k_means = KMeans(num_clusters=3, data_points=X)
clusters = k_means.fit(10)

plt.scatter(X[:, 0], X[:, 1], c=clusters)
plt.show()



