import pandas
from ml_algorithms import LinearRegression
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

# get training data
df = pandas.read_csv('data/train.csv', index_col='Id')

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
output.to_csv('submission.csv', index=False)