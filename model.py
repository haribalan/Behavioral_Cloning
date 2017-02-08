import csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import cv2
import matplotlib.pyplot as plt
from scipy import misc
from keras.models import Sequential,load_model
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

def roi(img): 
    img = img[60:140,40:280]
    return cv2.resize(img, (200, 66))

def preprocess_input(img):
    return roi(cv2.cvtColor(img, cv2.COLOR_RGB2YUV))
	

X_train = []
y_train = []
EPOCH = 10
with open('data/driving_log.csv') as csvfile:
	readCSV = csv.reader(csvfile, delimiter=',')
	next(readCSV,None)
	i=0
	for row in readCSV:
		file_name = row[0]
		try:
			if(file_name.index('Hari')):
				file_name = file_name.split('\\')[len(file_name.split('\\'))-1]
				file_name = 'IMG/'+file_name
		except:
			file_name = file_name
			
		img = misc.imread('data/'+file_name,mode='RGB')
		X_train.append(preprocess_input(img))
		y_train.append(row[3].strip())
		i+=1
	X_train = np.asarray(X_train,dtype=np.float32)
	y_train = np.asarray(y_train,dtype=np.float32)

X_train, y_train = shuffle(X_train, y_train)	

	
def normalize_grayscale(image_data):
    a = -0.5
    b = 0.5
    grayscale_min = 0
    grayscale_max = 255
    return a + ( ( (image_data - grayscale_min)*(b - a) )/( grayscale_max - grayscale_min ) )
	
X_normalized = normalize_grayscale(X_train)

try:
	model = load_model('my_model.h5')
	print('Model Loaded From Saved h5 file and used')
except:
	model = Sequential()

	model.add(Convolution2D(24, 5, 5,subsample=(2, 2), input_shape=(66,200,3),activation='relu')) #160, 320, 3)))

	model.add(Convolution2D(36, 5, 5,subsample=(2, 2),activation='relu'))

	model.add(Convolution2D(48, 5, 5,subsample=(2, 2),activation='relu'))

	model.add(Convolution2D(64, 3, 3,subsample=(1, 1),activation='relu'))

	model.add(Convolution2D(64, 3, 3,subsample=(1, 1),activation='relu'))

	model.add(Flatten())

	model.add(Dense(100,activation='relu'))

	model.add(Dense(50,activation='relu'))

	model.add(Dense(10,activation='relu'))

	model.add(Dense(1))

	model.compile('adam', 'mse', ['accuracy'])
	
	print('New Model is built')

history = model.fit(X_normalized, y_train, nb_epoch=EPOCH, validation_split=0.2)

model.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'

steering_angle = float(model.predict(X_normalized[1:2], batch_size=1))
print(y_train[1:2])
print(steering_angle)
