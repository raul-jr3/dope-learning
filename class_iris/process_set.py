import numpy as np
import pandas as pd 
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils

def load_iris_dataset(path_to_dataset):
	"""

	loads the iris dataset in the specified path
	and does the preprocessing before being fed
	into the neural net

	"""
	data = pd.read_csv(path_to_dataset)
	
	# drop the id column as it's not required -_-
	data = data[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm', 'Species']]

	# split the data into the input and it's labels
	train = data[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
	labels = data.Species

	encoder = LabelEncoder()
	encoder.fit(labels)
	encoded_labels = encoder.transform(labels)
	# one-hot-encode the labels
	labels_ohe = np_utils.to_categorical(encoded_labels, 3)

	train = np.array(train)
	labels_ohe = np.array(labels_ohe)

	return train, labels_ohe

def normalize_data(x):
	"""
	normalizes the data between an interval of [0, 1]

	"""
	x /= 255

	return x