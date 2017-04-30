import numpy as np
import csv
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import random
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.pooling import MaxPooling2D
from keras.layers.convolutional import Convolution2D
from keras import optimizers
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

## data generator function

def data_generator(image_files, batch_size):
	while 1:
		image_files = shuffle(image_files)
		for i in range(0, len(image_files), batch_size):
			X_data = []
			y_data = []
			batch_names = image_files[i:i+batch_size]
			for line in batch_names:
				filename = line[0]
				flipped = 0
				if "flipped" in filename:
					flipped = 1
					filename = filename.split('/')[0] + '/' + filename.split('/')[1] + '/' + filename.split('#')[1]

				img = plt.imread(filename)

				if flipped == 0:
					X_data.append(img)
				else:
					X_data.append(cv2.flip(img, 1))

				y_data.append(float(line[1]))

			X_data = np.array(X_data)
			y_data = np.array(y_data)
			yield (X_data, y_data)

flags = tf.app.flags
app_flags = flags.FLAGS

## model_name is the filename to save.

flags.DEFINE_string('model_name', '', "mode file name")
flags.DEFINE_integer('num_epochs', '', "num epochs")

images = [];
image_shape = (160, 320, 3)
model_name = app_flags.model_name
num_epochs = app_flags.num_epochs

### full_driving_log.csv combined and augmented training data from both tracks
### It is created by data_processing.ipynb

with open('full_driving_log.csv', 'r') as fb:
	lines = csv.reader(fb)
	for line in lines:
		if line[0] == 'center':
			continue

		if (("flipped" in line[0]) and (random.randint(0,1) == 0)):
			continue

		images.append(line)


batch_size = 128

## shuffle and split the training data.
train_images, validation_images = train_test_split(shuffle(images), train_size = 0.8)

## setup the generator functions for training and validation.
train_data_generator = data_generator(train_images, batch_size)
validation_data_generator = data_generator(validation_images, batch_size)
augmented_train_size = len(train_images);
augmented_valid_size = len(validation_images);

### Model
### Uses the NVIDIA architecture, normalization, cropping, and dropouts.

model = Sequential()
model.add(Lambda(lambda x: (x/255) - 0.5, input_shape = image_shape))
model.add(Cropping2D(cropping=((50, 20), (0,0)), input_shape = image_shape))
model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation = 'relu'))
model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation = 'relu'))
model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation = 'relu'))
model.add(Convolution2D(64, 3, 3, subsample=(2,2), activation = 'relu'))
model.add(Convolution2D(64, 3, 3, subsample=(1,1), activation = 'relu'))
model.add(Dropout(0.05))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.05))
model.add(Dense(50))
model.add(Dense(1))
adam_opt = optimizers.Adam(lr=0.001)
model.compile(loss='mse', optimizer=adam_opt)
model.fit_generator(train_data_generator, samples_per_epoch = augmented_train_size, nb_epoch = num_epochs, validation_data = validation_data_generator, nb_val_samples = augmented_valid_size)

## save the model using the name provided in the command line
model.save(model_name)
exit()
