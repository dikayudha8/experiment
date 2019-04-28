# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 11:43:21 2019

@author: dikayudha8
"""
import tensorflow as tf 
sess = tf.Session()
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import keras
import numpy as np

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, LSTM, TimeDistributed, Lambda
from keras.models import Model
from keras import backend as K

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

image_squences = 100

img_train = np.zeros(shape = (image_squences, 3,480, 640, 4))
img_raw = np.zeros(shape = (image_squences, 480, 640, 4))
img_raw_k = np.zeros(shape = (image_squences, 480, 640))
for i in range(1, image_squences + 1):
    for j in range(0,1):
        
        a = "left/tsukuba_daylight_L_" 
        disparity_name = "left_disparity/frame_" + str(i + j) + ".png"
        if i + j < 10: 
            b = "0000" + str(i + j) + ".png"
        elif i + j >= 10 and i + j < 100:
            b = "000" + str(i + j) + ".png"
        elif i + j >= 100 and i + j < 1000:
            b = "00" + str(i + j) + ".png"
        elif i + j >= 1000:
            b = "0" + str(i + j) + ".png"
    
        c = a + b

        img_raw[i - 1, :, :, :] = mpimg.imread(c)
        img_raw_k[i - 1, :, :] = mpimg.imread(disparity_name)
        img_train[i - 1, j, :, :, :] = mpimg.imread(c)
        
    
#import pandas as pd
#output_raw = pd.read_excel('D:/data/documents/project/DeepVO/odometry.xlsx')

input_img = Input(shape=(1, 480, 640, 4))  # the inputs are going to be 3 images with 640 x 480 each
x = TimeDistributed(Conv2D(1, (3, 3), activation='relu', padding='same'))(input_img)
x = TimeDistributed(MaxPooling2D((2, 2), padding='same'))(x)

x = TimeDistributed(Conv2D(1, (3, 3), activation='relu', padding='same'))(x)
x = TimeDistributed(MaxPooling2D((2, 2), padding='same'))(x)

x = TimeDistributed(Conv2D(1, (3, 3), activation='relu', padding='same'))(x)
x = TimeDistributed(MaxPooling2D((2, 2), padding='same'))(x)

x = TimeDistributed(Conv2D(1, (3, 3), activation='relu', padding='same'))(x)
x = TimeDistributed(MaxPooling2D((2, 2), padding='same'))(x)

x = TimeDistributed(Conv2D(1, (3, 3), activation='relu', padding='same'))(x)
x = TimeDistributed(MaxPooling2D((2, 2), padding='same'))(x)   

output_cnn = TimeDistributed(Flatten())(x)

rnn = LSTM(1000, return_sequences = 'true')(output_cnn)
rnn = LSTM(1000, return_sequences = 'true')(rnn)
# at this point the representation is (7, 7, 32)
R = Dense(12)(rnn)

input_img_enc = Input(shape=(480, 640, 4))  # adapt this if using `channels_first` image data format

y = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img_enc)
y = MaxPooling2D((2, 2), padding='same')(y)
y = Conv2D(8, (3, 3), activation='relu', padding='same')(y)
y = MaxPooling2D((2, 2), padding='same')(y)
y = Conv2D(8, (3, 3), activation='relu', padding='same')(y)
encoded = MaxPooling2D((2, 2), padding='same')(y)

# at this point the representation is (4, 4, 8) i.e. 128-dimensional

y = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
y = UpSampling2D((2, 2))(y)
y = Conv2D(8, (3, 3), activation='relu', padding='same')(y)
y = UpSampling2D((2, 2))(y)
y = Conv2D(16, (3, 3), activation='relu', padding='same')(y)
y = UpSampling2D((2, 2))(y)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(y)

   
def Warp_Images(x):
    fx = tf.to_float(tf.constant(615))
    fy = fx
    ox = tf.to_float(tf.constant(320))
    oy = tf.to_float(tf.constant(240))
    
    
    R = x[0]
    depth = x[1]    
    orig_image = x[2]
    
#    new_image = tf.Variable(np.zeros(shape = (image_squences, 480, 640, 4)))
#    print(R.shape)
#    for i in range(0, image_squences):
#        for y in range(0, 2):
#            for x in range(0, 2):
#                Z = depth[i, y, x]
#                X = ((x - ox)/fx)*Z
#                Y = ((y - oy)/fy)*Z
#                
#                X_p = R[i, 0, 0]*X + R[i, 0, 1]*Y + R[i, 0, 2]*Z + R[i, 0, 3]
#                Y_p = R[i, 0, 4]*X + R[i, 0, 5]*Y + R[i, 0, 6]*Z + R[i, 0, 7]
#                Z_p = R[i, 0, 8]*X + R[i, 0, 9]*Y + R[i, 0, 10]*Z + R[i, 0, 11]
#                
#                x_n = tf.to_int32(X_p * fx/Z_p + ox)
#                y_n = tf.to_int32(Y_p * fy/Z_p + oy)
#                
#                tf_i = tf.constant(i)
#                tf_color = tf.constant(0)
#                image_tf_idx = tf.Variable([tf_i, x_n, 1, tf_color])
#                
#                #one = tf.gather(orig_image, image_tf_idx)
#                
##                with sess.as_default():
##                    x_np = x_n.eval()
            
    return orig_image

img_raw_tensor = K.variable(img_raw)

x = img_raw_tensor

def Constructed_Images(x):
    width = K.variable(640.0)
    height = K.variable(480.0)
    
    fx = K.variable(615)
    fy = fx
    ox = K.variable(320)
    oy = K.variable(240)
    
    x_t, y_t = np.meshgrid(np.linspace(0, K.eval(width) - 1, K.eval(width)), np.linspace(0, K.eval(height) - 1, K.eval(height)))
    
    x_t = K.variable(x_t)
    y_t = K.variable(y_t)
    
    x_t_flat = K.reshape(x_t, (1, -1))
    y_t_flat = K.reshape(y_t, (1, -1))
    
    kstack = K.stack([image_squences, 1])
    x_t_flat = K.tile(x_t_flat, kstack)
    y_t_flat = K.tile(y_t_flat, kstack)
    
    x_t_flat = K.reshape(x_t_flat, [-1])
    y_t_flat = K.reshape(y_t_flat, [-1])
    
    #R = x[0]
    #depth = x[1]    
    orig_image = x[0]
    depth = x[1]
    pose = x[2]    
    
    x_0 = K.round(x_t_flat)
    y_0 = K.round(y_t_flat)
    
    x_0_d = K.cast(x_0, 'int32')
    y_0_d = K.cast(y_0, 'int32')
    
    Z_ = K.reshape(depth, np.stack([-1]))
    X_ = ((x_0_d - ox)/fx)*Z_
    Y_ = ((y_0_d - oy)/fy)*Z_    
    
    R_x = pose[:,:,0:4]
    R_y = pose[:,:,4:8]
    R_z = pose[:,:,7:12]
#    X_p = pose[0,0]*X_ + pose[1,0]*Y_ + pose[2,0]*Z_ + pose[3,0]
#    Y_p = pose[4,0]*X_ + pose[5,0]*Y_ + pose[6,0]*Z_ + pose[7,0]
#    Z_p = pose[8,0]*X_ + pose[9,0]*Y_ + pose[10,0]*Z_ + pose[11,0]
#    
#    x_n = np.clip((np.round(X_p * 615/Z_p + 320) + 19), 0, 639)
#    y_n = np.clip((np.round(Y_p * 615/Z_p + 240) + 14), 0, 479)
#    
#    x_n = x_n.astype(int)
#    y_n = y_n.astype(int)
    
    dim1 = orig_image.shape[1] * orig_image.shape[2]
    dim2 = orig_image.shape[2]
    
    def _repeat(x, n_repeats):
        rep = K.tile(K.expand_dims(x, 1), [1, n_repeats])
        
        return rep
    
    karange = K.arange(image_squences) * dim1
    base = _repeat(karange, height * width)
    base_y0 = base + y_0_d * dim2
    idx_l = base_y0 + x_0_d
    
    im_flat = K.reshape(orig_image, K.stack([-1, 4]))
    pix_l = K.gather(im_flat, idx_l)
    
    return pix_l
    

output_nn = Lambda(Constructed_Images, output_shape = (image_squences, 480, 640, 4))(decoded)


#test

def _np_repeat(x, n_repeats):
    rep = np.tile(np.expand_dims(x, 1), [1, n_repeats])
    
    return rep

x_t, y_t = np.meshgrid(np.linspace(0, 640 - 1, 640), np.linspace(0, 480 - 1, 480)) 

x_t_flat = np.reshape(x_t, (1, -1))
y_t_flat = np.reshape(y_t, (1, -1))

kstack = np.stack([image_squences, 1])
x_t_flat = np.tile(x_t_flat, kstack)
y_t_flat = np.tile(y_t_flat, kstack)

x_t_flat = np.reshape(x_t_flat, [-1])
y_t_flat = np.reshape(y_t_flat, [-1])
  
orig_image = img_raw
depth = img_raw_k

pose = np.array([1,0,0,0,0,1,0,0,0,0,0,1])
newPose = np.tile(pose, np.stack([image_squences, 1]))
newPose = np.reshape(newPose, (image_squences, 1, 12))

width = orig_image.shape[2]
height = orig_image.shape[1]

for i in range(0, image_squences):    
    Rx = np.tile(np.tile(newPose[i, 0, 0:4], np.stack([width * height, 1])), np.stack([image_squences, 1]))
    Ry = np.tile(np.tile(newPose[i, 0, 4:8], np.stack([width * height, 1])), np.stack([image_squences, 1]))
    Rz = np.tile(np.tile(newPose[i, 0, 8:12], np.stack([width * height, 1])), np.stack([image_squences, 1]))

x_0 = np.round(x_t_flat)
y_0 = np.round(y_t_flat)

x_0_d = x_0.astype(int)
y_0_d = y_0.astype(int)

Z_ = (615/643)*0.1/np.reshape(depth, np.stack([-1]))
X_ = ((x_0_d - 320)/615)*Z_
Y_ = ((y_0_d - 240)/615)*Z_

X_p = Rx[:,0]*X_ + Rx[:,1]*Y_ + Rx[:,2]*Z_ + Rx[:,3]
Y_p = Ry[:,0]*X_ + Ry[:,1]*Y_ + Ry[:,2]*Z_ + Ry[:,3]
Z_p = Rz[:,0]*X_ + Rz[:,1]*Y_ + Rz[:,2]*Z_ + Rz[:,3]

x_n = np.clip((np.round(X_p * 615/Z_p + 320) + 19), 0, 639)
y_n = np.clip((np.round(Y_p * 615/Z_p + 240) + 14), 0, 479)

x_n = x_n.astype(int)
y_n = y_n.astype(int)

dim1 = (orig_image.shape[1]) * (orig_image.shape[2])
dim2 = orig_image.shape[2]

karange = np.arange(image_squences) * dim1
base = _np_repeat(karange, 640 * 480)
base = np.reshape(base, [-1])
base_y0 = base + (y_n * dim2)
idx_l = base_y0 + x_n

maximumValTest = np.amax(idx_l)
im_flat = np.reshape(orig_image, np.stack([-1, 4]))
theImage = im_flat[idx_l]#, np.stack([480,640,4])

photo = np.reshape(theImage, np.stack([image_squences, 480,640,4]))
plt.imshow(photo[89,:,:,:])
    
    
    

    