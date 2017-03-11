import cv2,sys, getopt, json
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
from keras.models import Sequential,load_model
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from myutils import read_data_files, normalize_grayscale,trans_image,augment_brightness_camera_images

X_fname=[]
y_train = []
EPOCH = 25
DROPOUT = 0.3
BATCH_SIZE = 64 #28

X_fname, X_fname_val, y_train, y_validation = read_data_files()

def imgGen(files,angles):
		X_train = []
		index=0
		for item in files:
			x = misc.imread(item)
			y = angles[index]
			if(np.random.random()<.6):
				x, y = trans_image(x,y)
			if(np.random.random()>.5):
				x=augment_brightness_camera_images(x)
			random = np.random.randint(10)
			# Flip image in 50% of the cases
			# Thanks to Vivek Yadav for the idea
			if (random == 0):
				x = np.fliplr(x)
				y = -y
			X_train.append(x)
			angles[index]=y
			index+=1
		X_train = np.asarray(X_train,dtype=np.float32)
		return X_train,angles

def gen_batches(imgs, angles, batch_size,training):
	num_examples = len(imgs)
	start = 0
	end = batch_size
	while True:
		batch_x, batch_y = imgGen(imgs[start:end], angles[start:end])
		start+=batch_size
		end+=batch_size
		if start >= num_examples:
			start = 0
			end = batch_size
		yield batch_x, batch_y
		
try:
	if(len(sys.argv)!=2):
		print('Usage: python model.py <arg>')
		print('Where arg can be <filename>.h5 or lenet or nvidia')
		exit(-1)
	model_file = sys.argv[1]
	model_json = model_file.replace('h5','json')
	model = load_model(model_file)
	model.compile(optimizer = Adam(lr=0.001), loss='mean_squared_error', metrics=['accuracy'])
	print('Model Loaded From Saved h5 file and used: '+sys.argv[1])
except:
	print('New Model built: '+model_file)
	model = Sequential()
	if(model_file=='lenet'):
		model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
		model.add(Lambda(normalize_grayscale))
		model.add(Convolution2D(32, 3, 3, border_mode='valid'))
		model.add(MaxPooling2D(pool_size=(2, 2), border_mode='valid'))
		model.add(Dropout(DROPOUT))
		model.add(Activation('elu'))
		model.add(Convolution2D(32, 3, 3))
		model.add(MaxPooling2D(pool_size=(2, 2), border_mode='valid'))
		model.add(Dropout(DROPOUT))
		model.add(Activation('elu'))
		model.add(Flatten())
		model.add(Dense(128))
		model.add(Activation('elu'))
		model.add(Dense(43))
		model.add(Activation('elu'))
		model.add(Dense(10))
		model.add(Activation('elu'))
		model.add(Dense(1))
	elif(model_file=='nvidia'):
		model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
		model.add(Lambda(normalize_grayscale))
		model.add(Convolution2D(24, 5, 5,subsample=(2, 2))) #160, 320, 3)))
		model.add(Activation('elu'))
		model.add(Convolution2D(36, 5, 5,subsample=(2, 2)))
		model.add(Activation('elu'))
		model.add(Convolution2D(48, 5, 5,subsample=(2, 2)))
		model.add(Activation('elu'))
		model.add(Convolution2D(64, 3, 3,subsample=(1, 1)))
		model.add(Activation('elu'))
		model.add(Convolution2D(64, 3, 3,subsample=(1, 1)))
		model.add(Activation('elu'))
		model.add(Flatten())
		model.add(Dropout(DROPOUT))
		model.add(Dense(100))
		model.add(Activation('elu'))
		model.add(Dense(50))
		model.add(Activation('elu'))
		model.add(Dense(10))
		model.add(Activation('elu'))
		model.add(Dropout(DROPOUT))
		model.add(Dense(1))
	else:
		print('Check File Name. Note Usage: python model.py <arg>')
		print('Where arg can be <filename>.h5 or lenet or nvidia')
		exit(-1)
	
	model.compile(optimizer = Adam(lr=0.001), loss='mean_squared_error', metrics=['accuracy'])
	model_json=model_file+'_model.json'
	model_file=model_file+'_model.h5'
	
print(model.summary())
	

# Model will save the weights whenever validation loss improves
checkpoint = ModelCheckpoint(filepath = model_file, verbose = 1, save_best_only=True, monitor='val_loss')

# Discontinue training when validation loss fails to decrease
callback = EarlyStopping(monitor='val_loss', patience=5, verbose=1)

history_object = model.fit_generator(gen_batches(X_fname, y_train, BATCH_SIZE, True), len(X_fname),EPOCH,
	validation_data=gen_batches(X_fname_val, y_validation, BATCH_SIZE, False),nb_val_samples=len(X_fname_val),callbacks=[checkpoint, callback])

json_string = model.to_json()
with open(model_json, 'w') as jsonfile:
    json.dump(json_string, jsonfile)

#Print training and validation loss for each epoch 

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()


#Simple smoke test
#steering_angle = float(model.predict(imgGen(X_fname[1:2],y_train[1:2]), 1))
#print('y: %f  predict: %f'%(y_train[1:2],steering_angle))
