import csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import cv2,sys
import matplotlib.pyplot as plt
from scipy import misc
from keras.models import Sequential,load_model
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from myutils import preprocess_input


X_fname=[]
y_train = []
EPOCH = 5
DROPOUT = 0.2
BATCH_SIZE = 128 #28

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
		X_fname.append('data/'+file_name)	
		#img = misc.imread('data/'+file_name,mode='RGB')
		#X_train.append(preprocess_input(img))
		y_train.append(row[3].strip())
		i+=1
		#if(i==285):
		#	break
	#X_train = np.asarray(X_train,dtype=np.float32)
	X_fname = np.asarray(X_fname)
	y_train = np.asarray(y_train,dtype=np.float32)
	
X_fname, y_train = shuffle(X_fname, y_train)	

X_fname, X_fname_val, y_train, y_validation = train_test_split(X_fname, y_train, test_size=0.10) #, random_state=52)


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
		#if(offset>=num_examples):
		#	offset = 0
		#end = offset + batch_size
		#batch_x, batch_y = imgGen(imgs[offset:end]), angles[offset:end] #imgs[indeces], angles[indeces]
		#offset = end
		yield batch_x, batch_y
		
#import matplotlib.pyplot as plt
#plt.imshow(imgGen(['data/IMG/center_2016_12_01_13_35_29_104.jpg']).squeeze())
#plt.show()		
		
try:
	if(len(sys.argv)>1):
		model_file = sys.argv[1]
	model = load_model(model_file)
	print('Model Loaded From Saved h5 file and used')
except:
	model = Sequential()
	
	model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(66,200,3)))

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

	model.compile('adam', 'mse', ['accuracy'])
	
	print('New Model is built')

#history = model.fit(X_train, y_train, nb_epoch=EPOCH, validation_split=0.2)
history = model.fit_generator(gen_batches(X_fname, y_train, BATCH_SIZE), len(X_fname),EPOCH,validation_data=gen_batches(X_fname_val, y_validation, BATCH_SIZE),nb_val_samples=len(X_fname_val))

model.save('nvi_model.h5')  # creates a HDF5 file 'model.h5'

steering_angle = float(model.predict(imgGen(X_fname[1:2]), batch_size=1))
print('y: %f  predict: %f'%(y_train[1:2],steering_angle))
