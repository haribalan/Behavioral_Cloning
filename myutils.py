import cv2
import csv , random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

CORRECTION = 0.2
TRAIL = False
LEFT_RIGHT_CAM = True
#data_dir = '../../../Behavioral_Cloning/real/data/'
#data_dir = '../../../Behavioral_Cloning/real/data2/'
data_dir = '../../../Behavioral_Cloning/real/driving_data/'
	
def normalize_grayscale(image_data):
    return image_data/255.-0.5
	
def trans_image(image,steer,trans_range = 100):
    """
    Translation function provided by Vivek Yadav
    to augment the steering angles and images randomly
    and avoid overfitting
    """
    # Translation
    tr_x = trans_range*np.random.uniform()-trans_range/2
    steer_ang = steer + tr_x/trans_range*2*.2
    tr_y = 0
    Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])
    height, width = image.shape[:2]
    image_tr = cv2.warpAffine(image,Trans_M,(width,height))
    return image_tr,steer_ang

def augment_brightness_camera_images(image):
    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    image1 = np.array(image1, dtype = np.float64)
    random_bright = .5+np.random.uniform()
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1[:,:,2][image1[:,:,2]>255]  = 255
    image1 = np.array(image1, dtype = np.uint8)
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1
	
def cleanFileName(fname):
	file_name = fname.strip()
	try:
		if(file_name.index('Hari')):
			file_name = file_name.split('\\')[len(file_name.split('\\'))-1]
			file_name = 'IMG/'+file_name
	except:
		file_name = file_name
	return file_name
	
def read_data_files():
	X_fname=[]
	y_train = []
	with open(data_dir+'driving_log.csv') as csvfile:
		readCSV = csv.reader(csvfile, delimiter=',')
		next(readCSV,None)
		i=0
		for row in readCSV:
			if(float(row[3].strip())==0 and random.random()>0.7): #Skip zeros bias
				continue
			X_fname.append(data_dir+ cleanFileName(row[0]))	
			y_train.append(row[3].strip())
			#Since I also used beta simulator which does not create left and right images so handling it as separate case
			if LEFT_RIGHT_CAM:
				if(row[1].strip()!=''):
					X_fname.append(data_dir+cleanFileName(row[1]))
					y_train.append(str(float(row[3].strip())+CORRECTION))
				if(row[2].strip()!=''):
					X_fname.append(data_dir+cleanFileName(row[2]))
					y_train.append(str(float(row[3].strip())-CORRECTION))
			if TRAIL:
				i+=1
				if(i==100):
					break
		X_fname = np.asarray(X_fname)
		y_train = np.asarray(y_train,dtype=np.float32)
		
		#np.place(y_train, y_train>0.25, 0.2)
		#np.place(y_train, y_train<-0.25, -0.2)
		
		X_fname, y_train = shuffle(X_fname, y_train)	
	X_fname, X_fname_val, y_train, y_validation = train_test_split(X_fname, y_train, test_size=0.20) #, random_state=52)
	return X_fname, X_fname_val, y_train, y_validation