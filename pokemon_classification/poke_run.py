import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam
# the helpers
from process_data import *

# model parameters
n_hidden1 = 32
n_hidden2 = 64
nb_classes = 19
nb_epochs = 1750
optimizer = Adam()

def build_model():
	"""
	builds the neural network with three hidden layers
	with a relu activation function.
	The final output layer will have softmax as the
	activation function

	"""
	model = Sequential()
	model.add(Dense(nb_classes, input_shape = (8,)))
	model.add(Activation('relu'))
	model.add(Dense(n_hidden2))
	model.add(Activation('relu'))
	model.add(Dense(n_hidden2))
	model.add(Activation('relu'))
	model.add(Dense(n_hidden2))
	model.add(Activation('relu'))
	model.add(Dense(nb_classes))
	model.add(Activation('softmax'))

	return model

if __name__ == '__main__':

	# seed
	np.random.seed(1379)

	# get the dataset
	print("loading the dataset.........")
	train, labels = load_dataset('Pokemon.csv')
	train = normalize_poke_data(train)
	print("Dataset loaded!")
	# normalize the train data
	#train = normalize_poke_data(train)

	# build the neural network
	print("building the neural network.........")
	net = build_model()

	# compile the network
	net.compile(loss = 'categorical_crossentropy', optimizer = optimizer, metrics = ['accuracy'])

	# train the network
	print("training.........")
	net.fit(train, labels, epochs = nb_epochs, verbose = 1)

	score = net.evaluate(train, labels, verbose = 1)

	print("score :", score[0])
	print("accuracy :", score[1])
