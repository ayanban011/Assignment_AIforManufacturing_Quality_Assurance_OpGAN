import keras
import shutil
import keras
from keras.models import Model,Sequential
from keras.layers import *
import pandas as pd
import numpy as np
from keras.utils import to_categorical
from keras.models import Sequential
from keras.utils import np_utils
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.datasets import cifar10
from keras import regularizers
from keras.callbacks import LearningRateScheduler
import numpy as np
from keras.callbacks import ReduceLROnPlateau
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D
from keras.layers import Dropout, Flatten, GlobalAveragePooling2D
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage
from keras.callbacks import Callback,ModelCheckpoint
from keras.models import Sequential,load_model
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
import keras.backend as K
from sklearn.preprocessing import OneHotEncoder
import numpy as np # linear algebra
import pandas as pd 
from IPython.display import display, Image
from matplotlib.pyplot import imshow
from keras.layers import Conv2D, UpSampling2D, InputLayer
from keras.models import Sequential
from keras.preprocessing.image import img_to_array, load_img
from skimage.color import lab2rgb, rgb2lab
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from skimage.color import rgb2lab, lab2rgb, rgb2gray, xyz2lab
from skimage.io import imsave
import os
import random
from keras.layers.advanced_activations import*
from keras.optimizers import Adam
import PIL
from PIL import Image


def residual_layer(model):
  start = model
  forward = Conv2D(64, (3,3), padding='same', strides=1)(model)
  forward = BatchNormalization(momentum = 0.5)(forward)
  forward = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1,2])(forward)
  forward = Conv2D(64, (3,3), padding='same', strides=1)(forward)
  forward = BatchNormalization(momentum = 0.5)(forward)
  
  residual = add([start,forward])
  
  return residual


def upsample(model):
  upscale = Conv2D(64, (3,3), padding='same', strides=1)(model)
  upscale = UpSampling2D()(upscale)
  upscale = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1,2])(upscale)
  
  return upscale


def generator(res_size, up_size, shape):
  gen_inp = Input(shape = shape)
  model = Conv2D(64, (9,9), padding='same', strides=1)(gen_inp)
  model = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1,2])(model)
  
  start_res = model
  
  for i in range(res_size):
    model = residual_layer(model)
    
  model = Conv2D(64, (3,3), padding='same', strides=1)(model)
  model = BatchNormalization(momentum = 0.5)(model)
  
  model = add([start_res,model])
  
  
  for i in range(up_size):
    model = upsample(model)
    
  final = Conv2D(3, (9,9), activation = 'tanh', padding='same', strides=1)(model)
  
  gen = Model(inputs=gen_inp, outputs=final)
  
  return gen


def dis_layer(model, filters, kernal_size, stride):
  model = Conv2D(filters = filters, kernel_size = kernal_size, padding='same', strides= stride)(model)
  model = BatchNormalization(momentum=0.5)(model)
  model = LeakyReLU(alpha = 0.2)(model)
  
  return model

def discriminator(shape):
  dis_inp = Input(shape = shape)
  model = Conv2D(64, kernel_size = 3, padding='same', strides=1)(dis_inp)
  model = BatchNormalization(momentum=0.5)(model)
  
  model = dis_layer(model, 64, 3, 1)
  model = dis_layer(model, 128, 3, 1)
  model = dis_layer(model, 128, 3, 2)
  model = dis_layer(model, 256, 3, 2)
  model = dis_layer(model, 256, 3, 2)
  model = dis_layer(model, 512, 3, 2)
  model = dis_layer(model, 512, 3, 2)
  
  model = Flatten()(model)
  model = Dense(100)(model)
  model = LeakyReLU(alpha = 0.2)(model)
  model = Dense(1, activation = 'sigmoid')(model)
  
  dis = Model(inputs=dis_inp, outputs=model)
  return dis

from keras.applications.vgg19 import VGG19
def vgg_loss(true, pred):
  vgg = VGG19(include_top=False, weights='imagenet')
  vgg.trainable = False
  model = Model(inputs = vgg.input, outputs = vgg.get_layer('block5_conv4').output)
  
  model.trainable = False
  
  return K.mean( K.square( model(true) - model(pred) ) )

files = os.listdir('train2017')

x_hr = np.empty((800,384,384,3), 'float64')
x_lr = np.empty((800,96,96,3), 'float64')

for i in range(800):
  image = Image.open('train2017/'+files[4*i])
  image1 = image.resize((384,384), resample=PIL.Image.BICUBIC)
  image1 = np.array(image1)
  if image1.shape ==  (384,384):
    image = Image.open('train2017/'+files[4*i+1])
    image1 = image.resize((384,384), resample=PIL.Image.BICUBIC)
    image1 = np.array(image1)
    x_hr[i] = image1
    im = image.resize((96,96), resample=PIL.Image.BICUBIC)
    im = np.array(im)
    x_lr[i] = im
  else:
    x_hr[i] = image1
    im = image.resize((96,96), resample=PIL.Image.BICUBIC)
    im = np.array(im)
    x_lr[i] = im
  
  
x_hr = (x_hr - 127.5)/127.5
x_lr = (x_lr - 127.5)/127.5

gen_shape = (96,96,3)
dis_shape = (384,384,3)

adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

generator = generator(16, 2, gen_shape)

discriminator = discriminator(dis_shape)
discriminator.compile(loss="binary_crossentropy", optimizer=adam)

discriminator.trainable = False

gan_input = Input(shape = gen_shape)
gen_out = generator(gan_input)
gan_final = discriminator(gen_out) 
gans = Model(inputs=gan_input, outputs=[gen_out,gan_final])
gans.compile(loss=[vgg_loss, "binary_crossentropy"], loss_weights=[1., 1e-3], optimizer=adam)




tt1=[]
tt2=[]
m = x_hr.shape[0]
loss_history = []
batch_size = 4
for epoch in range(31):
    itera  = int(m/batch_size)
    dis_mean = 0
    gan_mean = 0
    for i in range(itera):
      high_resol = x_hr[i*batch_size:min((i+1)*batch_size,m)]
      low_resol = x_lr[i*batch_size:min((i+1)*batch_size,m)]
      
      upscale_img = generator.predict(low_resol)
      
      real = np.ones(high_resol.shape[0]) - np.random.random_sample(high_resol.shape[0])*0.1
      fake = np.random.random_sample(low_resol.shape[0])*0.1
      
      dis_loss1 = discriminator.train_on_batch(x = high_resol,
                                         y = real)
      dis_loss2 = discriminator.train_on_batch(x = upscale_img,
                                    y = fake)
      
      dis_loss = (dis_loss1 + dis_loss2)*0.5
      
      dis_mean = dis_mean + dis_loss
      
      gan_loss = gans.train_on_batch(x = low_resol,
                                     y = [high_resol, real])
      gan_loss = gan_loss[0] + gan_loss[1]*1e-3
      
      gan_mean = gan_mean + gan_loss
      
      
      print('Epoch = '+str(epoch)+' batch = '+str(i)+' | discriminator loss = '+str(dis_loss)+' | gan loss = '+str(gan_loss))
    
    dis_mean = dis_mean/itera
    gan_mean = gan_mean/itera
    print('Epoch = '+str(epoch)+' | mean discriminator loss = '+str(dis_mean)+' | mean gan loss = '+str(gan_mean))
    tt1.append(dis_mean)
    tt2.append(gan_mean)
    print('------------------------------------------------Epoch '+str(epoch)+' complete-----------------------------------------------')

