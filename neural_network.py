import numpy as np
import keras
from keras.datasets import mnist

attrNum = 4
numClss = 3
numHidden = 10
l = 0.1
lookup = np.array(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])
filename = "dataset/iris.csv"
sep = ','
numEpochs = 10000
batchSize = 8


def sigmoid(x, deriv=False):

    if(deriv == True):
        return  x * (1 - x)

    return 1 / (1 + np.exp(-x))


# Normal distribution initialization with sigma=0.4 and mean=0
syn0 = np.random.normal(0, 0.4, (attrNum,numHidden))
syn1 = np.random.normal(0, 0.4, (numHidden,numClss))

k = 0
countBatch = 0

# data from handwritten digits MNIST dataset
'''(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
x_train = x_train.reshape(60000, 28*28)
x_test = x_test.reshape(10000, 28*28)

y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)'''

while k < numEpochs:

    index = 0
    l2_delta = np.zeros([syn1.shape[1]])
    l1_delta = np.zeros([syn0.shape[1]])
    countBatch = 0
    x_train = []
    y_train = []

    # training from csv data
    file = open(filename, "r")

    for line in file:

        instance = np.zeros([attrNum])
        q = 0
        line = line.strip('\n')

        for word in line.split(sep):

            if q == attrNum:
                break

            instance[q] = float(word)
            q += 1

        classes_vect = np.zeros([numClss])
        cls = np.argwhere(lookup == word)[0]
        classes_vect[cls] = 1
        x_train.append(instance)
        y_train.append(classes_vect)

    for instance, classes_vect in zip(x_train, y_train):
        l0 = instance
        l1 = sigmoid(np.dot(l0, syn0))
        l2 = sigmoid(np.dot(l1, syn1))

        l2_error = classes_vect - l2
        l2_delta += l2_error * sigmoid(l2, deriv=True)
        l1_error = l2_delta.dot(syn1.T)
        l1_delta += l1_error * sigmoid(l1, deriv=True)
        countBatch += 1

        if countBatch == batchSize:
            for j in range(0, syn1.shape[1]):
                for i in range(0, syn1.shape[0]):
                    syn1[i][j] += l * l2_delta[j] / batchSize * l1[i]

            for j in range(0, syn0.shape[1]):
                for i in range(0, syn0.shape[0]):
                    syn0[i][j] += l * l1_delta[j] / batchSize * l0[i]

            countBatch = 0
            l2_delta = np.zeros([syn1.shape[1]])
            l1_delta = np.zeros([syn0.shape[1]])

        index += 1

    k += 1

    print('{}/{}'.format(k, numEpochs))

    '''corrects = 0.0
    errors = 0.0

    for line in file:

	    instance = np.zeros([attrNum])

	    q = 0

	    line = line.strip('\n')

	    for word in line.split(sep):

		if q == attrNum:
		    break

		instance[q] = float(word)

		q += 1

    for instance, cls in zip(x_train, y_train):

        l0 = instance
        l1 = sigmoid(np.dot(l0, syn0))
        l2 = sigmoid(np.dot(l1, syn1))
        predicted = np.argmax(l2)
        cls = np.argwhere(lookup == word)[0]

        #cls = np.argmax(cls)

        if predicted == cls:
            corrects += 1
        else:
            errors += 1

    print('{}/{} | error: {}'.format(k, numEpochs, corrects / (corrects + errors)))

    file.close()

    file = open(filename, "r")

    corrects = 0.0
    errors = 0.0

    for line in file:

        instance = np.zeros([attrNum])

        q = 0

        line = line.strip('\n')

        for word in line.split(sep):

            if q == attrNum:
                break

            instance[q] = float(word)

            q += 1

    for instance, cls in zip(x_test, y_test):

        l0 = instance
        l1 = sigmoid(np.dot(l0, syn0))
        l2 = sigmoid(np.dot(l1, syn1))
        predicted = np.argmax(l2)
        #cls = np.argwhere(lookup == word)[0]

        cls = np.argmax(cls)

        if predicted == cls:
            corrects += 1
        else:
            errors += 1

    print("Error:")
    print((corrects / (corrects + errors)))'''

file.close()

file = open(filename, "r")

corrects = 0.0
errors = 0.0

confusion_matrix = np.zeros([numClss, numClss])

for line in file:

    instance = np.zeros([attrNum])

    q = 0

    line = line.strip('\n')

    for word in line.split(sep):

        if q == attrNum:
            break

        instance[q] = float(word)

        q += 1

    l0 = instance
    l1 = sigmoid(np.dot(l0, syn0))
    l2 = sigmoid(np.dot(l1, syn1))
    predicted = np.argmax(l2)
    cls = np.argwhere(lookup == word)[0]

    if predicted == cls:
        corrects += 1
    else:
        errors += 1

    confusion_matrix[predicted][cls] += 1

    k += 1

print("Error:")
print((corrects / (corrects + errors)))
print("Confusion matrix:")
for i in range(0, numClss):
    print(confusion_matrix[i])

