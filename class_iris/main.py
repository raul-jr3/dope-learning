import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import Adam, SGD
import process_set # helpers

# model params
# DROPOUT = 0.2
nb_classes = 3 # three categories
nb_epochs = 250
optimizer = Adam()
batch_size = 15
n_hidden = 32

def build_neural_network():
	"""
	the neural network will have two hidden layers
	with a relu activation for each.
	The final output layer will have the softmax function

	"""
	model = Sequential()
	model.add(Dense(nb_classes, input_shape = (4,)))
	model.add(Activation('relu'))
	model.add(Dense(n_hidden))
	model.add(Activation('relu'))
	model.add(Dense(nb_classes))
	model.add(Activation('softmax'))

	return model

if __name__ == '__main__':

	np.random.seed(1379)

	train, labels = process_set.load_iris_dataset('Iris.csv')
	# normalize the dataset
	train = process_set.normalize_data(train)
	print("dataset loaded and normalized")
	print("building the neural network.........")

	neural_net = build_neural_network()
	print("neural network built!!!")

	print("now for the compilation :P")
	print("compiling")
	# compile the model after it is built
	neural_net.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy'])

	print("training the neural network")
	# train the neural network
	neural_net.fit(train, labels, batch_size = batch_size, epochs = nb_epochs, verbose = 1)

	score = neural_net.evaluate(train, labels, verbose = 1)

	print("score :", score[0])
	print("accuracy :", score[1])