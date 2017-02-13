import cv2,sys, getopt
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
from keras.models import Sequential,load_model
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam
from myutils import preprocess_input, read_data_files, normalize_grayscale

X_fname=[]
y_train = []
EPOCH = 20
DROPOUT = 0.2
BATCH_SIZE = 128 #28

X_fname, X_fname_val, y_train, y_validation = read_data_files()

def imgGen(files):
		X_train = []
		for i in files:
			img = misc.imread(i,mode='RGB')
			X_train.append(preprocess_input(img))
		X_train = np.asarray(X_train,dtype=np.float32)
		return X_train

def gen_batches(imgs, angles, batch_size):
	num_examples = len(imgs)
	offset = 0
	while True:
		indeces = np.random.choice(num_examples, batch_size)
		batch_x, batch_y = imgGen(imgs[indeces]), angles[indeces]
		yield batch_x, batch_y
		
try:
	if(len(sys.argv)!=2):
		print('Usage: python model.py <arg>')
		print('Where arg can be <filename>.h5 or lenet or nvidia')
		exit(-1)
	model_file = sys.argv[1]
	model = load_model(model_file)
	print('Model Loaded From Saved h5 file and used: '+sys.argv[1])
except:
	print('New Model built: '+model_file)
	model = Sequential()
	if(model_file=='lenet'):
		model.add(Lambda(normalize_grayscale, input_shape=(66,200,3)))
		model.add(Convolution2D(32, 3, 3, border_mode='valid'))
		model.add(MaxPooling2D(pool_size=(2, 2), border_mode='valid'))
		model.add(Dropout(DROPOUT))
		model.add(Activation('relu'))
		model.add(Convolution2D(32, 3, 3))
		model.add(MaxPooling2D(pool_size=(2, 2), border_mode='valid'))
		model.add(Dropout(DROPOUT))
		model.add(Activation('relu'))
		model.add(Flatten())
		model.add(Dense(128))
		model.add(Activation('relu'))
		model.add(Dense(43))
		model.add(Activation('relu'))
		model.add(Dense(10,activation='relu'))
		model.add(Dense(1))
	elif(model_file=='nvidia'):
		model.add(Lambda(normalize_grayscale, input_shape=(66,200,3)))
		model.add(Convolution2D(24, 5, 5,subsample=(2, 2),activation='relu')) #160, 320, 3)))
		model.add(Convolution2D(36, 5, 5,subsample=(2, 2),activation='relu'))
		model.add(Convolution2D(48, 5, 5,subsample=(2, 2),activation='relu'))
		model.add(Dropout(DROPOUT))
		model.add(Convolution2D(64, 3, 3,subsample=(1, 1),activation='relu'))
		model.add(Convolution2D(64, 3, 3,subsample=(1, 1),activation='relu'))
		#model.add(Dropout(DROPOUT))
		model.add(Flatten())
		model.add(Dense(100,activation='relu'))
		model.add(Dense(50,activation='relu'))
		model.add(Dense(10,activation='relu'))
		model.add(Dense(1))
	else:
		print('Check File Name. Note Usage: python model.py <arg>')
		print('Where arg can be <filename>.h5 or lenet or nvidia')
		exit(-1)
	
	model.compile(optimizer = Adam(lr=0.001), loss='mean_squared_error', metrics=['accuracy'])
	model_file=model_file+'_model.h5'
	
history = model.fit_generator(gen_batches(X_fname, y_train, BATCH_SIZE), len(X_fname),EPOCH,validation_data=gen_batches(X_fname_val, y_validation, BATCH_SIZE),nb_val_samples=len(X_fname_val))
model.save(model_file)  # creates a HDF5 file 'model.h5'

#Simple smoke test
steering_angle = float(model.predict(imgGen(X_fname[1:2]), batch_size=1))
print('y: %f  predict: %f'%(y_train[1:2],steering_angle))
