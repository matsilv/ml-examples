import pandas
from ml_algorithms import LinearRegression, LogisticRegression, NeuralNetwork
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn.impute import SimpleImputer


# get training data
'''df = pandas.read_csv('data/train.csv', index_col='Id')

Y = df['SalePrice'].copy()
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = df[features]
x = X.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
X = pandas.DataFrame(x_scaled)

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# create and train the model
model = LinearRegression(num_features=len(features), lr=0.1)
history = model.train(X=x_train, Y=y_train, epochs=2000)
model.test(X=x_test, Y=y_test)

history = history[100:]
plt.plot(np.arange(0, len(history)), history)
plt.show()

# get training data
df = pandas.read_csv('data/test.csv', index_col='Id')

X = df[features]
x = X.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
X = pandas.DataFrame(x_scaled)

predictions = model.predict(X)
# save test predictions to file
output = pandas.DataFrame({'Id': df.index,
                       'SalePrice': predictions.reshape(-1)})

# file submission for Kaggle competition
output.to_csv('submission.csv', index=False)'''

df = pandas.read_csv('data/iris.csv')
df = df.replace('Iris-setosa', 1)
df = df.replace('Iris-versicolor', 2)
df = df.replace('Iris-virginica', 3)

features = df.columns[:-1]
X = df[features].copy()
Y = df['E'].copy()

Y = pandas.get_dummies(Y)

my_imputer = SimpleImputer()
X = pandas.DataFrame(my_imputer.fit_transform(X))

'''x = X.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
X = pandas.DataFrame(x_scaled)'''

#x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.0, random_state=42)

'''model = LogisticRegression(num_features=4, lr=0.01)
history = model.train(X=x_train, Y=y_train, epochs=3000, batch_size=64)
model.test(X=x_test, Y=y_test)'''

model = NeuralNetwork(attr_num=4, num_clss=3, num_hidden=10)
history = model.train(num_epochs=100, lr=0.1, batch_size=8, X=X, y=Y)

history = history[:]
plt.plot(np.arange(0, len(history)), history)
plt.show()

'''df = pandas.read_csv('data/titanic/test.csv', index_col='PassengerId')

X = df[features]
x = X.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
X = pandas.DataFrame(x_scaled)

predictions = model.predict(X)
# save test predictions to file
output = pandas.DataFrame({'PassengerId': df.index,
                       'Survived': predictions.reshape(-1)})

# file submission for Kaggle competition
output.to_csv('submission.csv', index=False)'''