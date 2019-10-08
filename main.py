import pandas
from ml_algorithms import LinearRegression
from sklearn import preprocessing

# get training data and classes
df = pandas.read_csv('data/train.csv')


y_train = df['SalePrice'].copy()
features = ['LotArea', 'YearBuilt']
x_train = df[features]
x = x_train.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
x_train = pandas.DataFrame(x_scaled)

# create and train the model
model = LinearRegression(num_features=len(features))
model.train(X=x_train, Y=y_train, epochs=10000)
