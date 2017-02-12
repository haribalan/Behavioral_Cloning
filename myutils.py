import cv2

def roi(img): 
    img = img[60:140,40:280]
    return cv2.resize(img, (200, 66))

def preprocess_input(img):
    return roi(img) 
	#return roi(cv2.cvtColor(img, cv2.COLOR_RGB2YUV))
	
	
def normalize_grayscale(image_data):
    a = -0.5
    b = 0.5
    grayscale_min = 0
    grayscale_max = 255
    return image_data/255.0 - 0.5 
	#return a + ( ( (image_data - grayscale_min)*(b - a) )/( grayscale_max - grayscale_min ) )
	