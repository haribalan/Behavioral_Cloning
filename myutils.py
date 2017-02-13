import cv2
import csv 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

CORRECTION = 0.18
TRAIL = False
LEFT_RIGHT_CAM = False
MOVING_AVG_WINDOW = 3
def roi(img): 
    img = img[60:140,40:280]
    return cv2.resize(img, (200, 66))

def preprocess_input(img):
    return roi(img) 
	#return roi(cv2.cvtColor(img, cv2.COLOR_RGB2YUV))
	
	
def normalize_grayscale(image_data):
    return image_data/255.0 - 0.5 

	
def moving_average(a,n):
	'''
	Calculate moving average
	http://stackoverflow.com/a/14314054
	'''
	if n==1:
		return a
	ret = np.cumsum(a, dtype=float)
	ret[n:] = ret[n:] - ret[:-n]
	return ret[n - 1:] / n
	
def read_data_files():
	X_fname=[]
	y_train = []
	with open('data/driving_log.csv') as csvfile:
		readCSV = csv.reader(csvfile, delimiter=',')
		next(readCSV,None)
		i=0
		for row in readCSV:
			file_name = row[0].strip()
			try:
				if(file_name.index('Hari')):
					file_name = file_name.split('\\')[len(file_name.split('\\'))-1]
					file_name = 'IMG/'+file_name
			except:
				file_name = file_name
			X_fname.append('data/'+file_name)	
			y_train.append(row[3].strip())
			#Since I use beta simulator which does not create left and right images so handling it as separate case
			if LEFT_RIGHT_CAM:
				if(row[1].strip()!=''):
					X_fname.append('data/'+row[1].strip())
					y_train.append(str(float(row[3].strip())+CORRECTION))
				if(row[2].strip()!=''):
					X_fname.append('data/'+row[2].strip())
					y_train.append(str(float(row[3].strip())-CORRECTION))
			if TRAIL:
				i+=1
				if(i==300):
					break
		X_fname = np.asarray(X_fname)
		y_train = np.asarray(y_train,dtype=np.float32)

	#y_train = moving_average(y_train,MOVING_AVG_WINDOW)
	#X_fname = X_fname[(MOVING_AVG_WINDOW - 1):] #[3 * (MOVING_AVG_WINDOW - 1):]

	X_fname, y_train = shuffle(X_fname, y_train)	

	X_fname, X_fname_val, y_train, y_validation = train_test_split(X_fname, y_train, test_size=0.20) #, random_state=52)

	return X_fname, X_fname_val, y_train, y_validation