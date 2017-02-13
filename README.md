# Behavioral_Cloning

## Introduction

The goal of the project is to performs behavioral cloning, which involves training an deep neural network to mimic human driving behavior in a simulator by predicting Steering Angles from Camera Images.

### Simulator
Training data was generated using the beta simulator for this project. Beta simulator uses the mice to drive the car around track providing more smoother angles than the Stable simulator which uses Keyboard as input device.

### Data collection
Training data is collected by a manually driving around a track in the simulator. To train car to move away from the sides during autonomous mode and stay in middle of the lane and also avoid overfitting: explicit recording of pulling back into the middle of the lane was performed both from left, right and on deep curves. 

Model was then trained on data collected using the vehicle's camera images collected from the manual demonstration. The final trained model is tested against the same test track by driving the car on autonomous mode. 

#### Project Dependency
*	Udacity behavioral cloning simulator
*	AWS GPU (and CPU pc) used for training 
*	Carnd-Term1 requirements such as:
    *	Python 3.5
    *	TensorFlow
    *	Keras
    *	PIL
    *	Numpy
    *	h5py
    *	Scikit Learn

#### Submission
* model.py – This has both LENET^ and NVIDIA* model. It can run both based on the inputs key (lenet or nvidia)
	* *https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
	* ^http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf
	*	Run:  python model.py lenet  [or]  python model.py nvidia
*	drive.py - The script to load the trained model, interface with the simulator, and drive the car
	*	Run: python driver.py <nvidia/lenet>_model.h5
*	<nvidia/lenet>_model.h5 - The model weights
	*	myutils.py – utility methods

### Data Processing

##### Images processing and normalization

Images were cropped to retain only the area of interest by removing top portion (up to middle of the image) and bottom (very close to hood of the car) areas and primarily have only the road/lane section. 

![alt text](images/full_img.PNG"Full Image Before Crop")
![alt text](images/crop_img.png"Image After Crop")

Images were then resized to 200*66 and then input into the deep learning model. On the deep learning model the first learn is Normalization which reduces the scale from 0 – 255 to -0.5 – 0.5 using Lambda layer.

Images are then changed to COLOR_RGB2YUV mode, for better performance.

###### Angles

Moving average where n=3 was performed to smooth out the angles.

### Model Architecture

##### My model based on Nvidia
```
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to
====================================================================================================
lambda_1 (Lambda)                (None, 66, 200, 3)    0           lambda_input_1[0][0]
____________________________________________________________________________________________________
convolution2d_1 (Convolution2D)  (None, 31, 98, 24)    1824        lambda_1[0][0]
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 14, 47, 36)    21636       convolution2d_1[0][0]
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 5, 22, 48)     43248       convolution2d_2[0][0]
____________________________________________________________________________________________________
convolution2d_4 (Convolution2D)  (None, 3, 20, 64)     27712       convolution2d_3[0][0]
____________________________________________________________________________________________________
convolution2d_5 (Convolution2D)  (None, 1, 18, 64)     36928       convolution2d_4[0][0]
____________________________________________________________________________________________________
dropout_1 (Dropout)              (None, 1, 18, 64)     0           convolution2d_5[0][0]
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 1152)          0           dropout_1[0][0]
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 100)           115300      flatten_1[0][0]
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 50)            5050        dense_1[0][0]
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 10)            510         dense_2[0][0]
____________________________________________________________________________________________________
dense_4 (Dense)                  (None, 1)             11          dense_3[0][0]
====================================================================================================
Total params: 252,219
Trainable params: 252,219
Non-trainable params: 0
```
##### My Model based on LeNet:
```
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to
====================================================================================================
lambda_1 (Lambda)                (None, 66, 200, 3)    0           lambda_input_1[0][0]
____________________________________________________________________________________________________
convolution2d_1 (Convolution2D)  (None, 64, 198, 32)   896         lambda_1[0][0]
____________________________________________________________________________________________________
maxpooling2d_1 (MaxPooling2D)    (None, 32, 99, 32)    0           convolution2d_1[0][0]
____________________________________________________________________________________________________
dropout_1 (Dropout)              (None, 32, 99, 32)    0           maxpooling2d_1[0][0]
____________________________________________________________________________________________________
activation_1 (Activation)        (None, 32, 99, 32)    0           dropout_1[0][0]
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 30, 97, 32)    9248        activation_1[0][0]
____________________________________________________________________________________________________
maxpooling2d_2 (MaxPooling2D)    (None, 15, 48, 32)    0           convolution2d_2[0][0]
____________________________________________________________________________________________________
dropout_2 (Dropout)              (None, 15, 48, 32)    0           maxpooling2d_2[0][0]
____________________________________________________________________________________________________
activation_2 (Activation)        (None, 15, 48, 32)    0           dropout_2[0][0]
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 23040)         0           activation_2[0][0]
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 128)           2949248     flatten_1[0][0]
____________________________________________________________________________________________________
activation_3 (Activation)        (None, 128)           0           dense_1[0][0]
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 43)            5547        activation_3[0][0]
____________________________________________________________________________________________________
activation_4 (Activation)        (None, 43)            0           dense_2[0][0]
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 10)            440         activation_4[0][0]
____________________________________________________________________________________________________
dense_4 (Dense)                  (None, 1)             11          dense_3[0][0]
====================================================================================================
Total params: 2,965,390
Trainable params: 2,965,390
Non-trainable params: 0
```




