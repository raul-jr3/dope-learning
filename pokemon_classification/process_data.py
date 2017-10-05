import numpy as np 
import pandas as pd 
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder

def load_dataset(path_to_data):
	"""
	loads the dataset specified
	and splits it into the input and labels
	and the labels are one-hot-encoded

	"""
	data = pd.read_csv(path_to_data)

	# use the important columns and drop the rest
	data = data[['Total', 'HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed', 'Generation', 'Type 1']]

	# now the data is split into the train and labels
	train = data[['Total', 'HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed', 'Generation']]
	labels = data['Type 1']

	# convert the labels into one-hot-encode vectors
	encoder = LabelEncoder()
	encoder.fit(labels)
	encoded_labels = encoder.transform(labels)

	# one-hot-encode using keras' to_categorical
	ohe_labels = np_utils.to_categorical(encoded_labels, 19)

	train = np.array(train, dtype = 'float32')
	#train = train.reshape(800, 8)
	ohe_labels = np.array(ohe_labels, dtype = 'float32')

	return train, ohe_labels

def normalize_poke_data(x):

	x /= 255

	return x