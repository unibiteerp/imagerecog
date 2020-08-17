# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 15:55:24 2020

@author: Preeti
"""
import os    
import cv2
import tensorflow
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import array_to_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
import pickle
import pydot

from numpy.random import seed
seed(10)
#tensorflow.random.set_seed(20)

datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.05,    #0.2
        height_shift_range=0.05,   #0.2
        shear_range=0.05,          # 0.2
        zoom_range=0.1,            #0.2
        horizontal_flip=True,
        fill_mode='nearest')

# Resize images
path_of_images = r"C:\Users\Preeti\Codes\ImageApp\dataset"
folders_in_path = os.listdir(path_of_images)
list_of_images = os.listdir(path_of_images)
for folder in folders_in_path:
    path_of_images = r"C:\Users\Preeti\Codes\ImageApp\dataset" + '\\' + folder
    list_of_images = os.listdir(path_of_images)
    for image in list_of_images:
        img = cv2.imread(os.path.join(path_of_images, image))
        #Resize the image
        basewidth = 300
        hsize = 300
        img = Image.open(path_of_images +'\\'+image)
        img = img.resize((basewidth,hsize), Image.ANTIALIAS)
        img.save(path_of_images +'\\'+image) 
        x = img_to_array(img)  # this is a Numpy array with shape (300, 300, 3)
        x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 300, 300, 3)
        x.shape

    
# the .flow() command below generates batches of randomly transformed images
# and saves the results to the `dataset/` directory
        i = 0
        for batch in datagen.flow(x, batch_size=2,
                                 save_to_dir='dataset'+'\\'+folder , save_prefix= folder, save_format='jpeg'):
            i += 1
            if i > 20:
                break  # otherwise the generator would loop indefinitely

# Train, validation split
import glob
import os
from random import shuffle
import shutil
path_of_images = r"C:\Users\Preeti\Codes\ImageApp\dataset"
folders_in_path = os.listdir(path_of_images)
train_filenames = []
val_filenames = []
#test_filenames = []
for folder in folders_in_path:
    path_of_images = r"C:\Users\Preeti\Codes\ImageApp\dataset" + '\\' + folder
    files = list(glob.glob(os.path.join(path_of_images,'*.*')))
    shuffle(files)
    split_1 = int(0.8 * len(files))
    #split_2 = int(0.9 * len(files))
    train_filenames = train_filenames + files[:split_1]
    val_filenames = val_filenames + files[split_1:]
    #test_filenames = test_filenames + files[split_2:]

for i in train_filenames:
        if (os.path.isfile(i)) and ("tapir" in i):
            destpath = "C:\\Users\\Preeti\\Codes\\ImageApp\\train\\tapir"
            shutil.copy(i,destpath)
        else:
            destpath = "C:\\Users\\Preeti\\Codes\\ImageApp\\train\\elephant"
            shutil.copy(i,destpath)
            
for i in val_filenames:
        if (os.path.isfile(i)) and ("tapir" in i):
            destpath = "C:\\Users\\Preeti\\Codes\\ImageApp\\validation\\tapir"
            shutil.copy(i,destpath)
        else:
            destpath = "C:\\Users\\Preeti\\Codes\\ImageApp\\validation\\elephant"
            shutil.copy(i,destpath)
            
#https://www.learnopencv.com/number-of-parameters-and-tensor-sizes-in-convolutional-neural-network/
#https://www.learnopencv.com/batch-normalization-in-deep-networks/
# MODEL
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization

from keras.utils import plot_model
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(300, 300, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())  #the initial values of our parameters by initializing them with zero mean and unit variance
#model.add(Dropout(0.25))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))
model.add(BatchNormalization())

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))
model.add(BatchNormalization())

# the model so far outputs 3D feature maps (height, width, features)

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(1))
model.add(Activation('sigmoid'))
# COMPILE
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',     #adam
              metrics=['accuracy'])



#https://towardsdatascience.com/understanding-binary-cross-entropy-log-loss-a-visual-explanation-a3ac6025181a

batch_size = 20

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.05,  #0.2
        zoom_range=0.1,  #0.2
        horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1./255)

# this is a generator that will read pictures found in subfolers of 'data/train', and indefinitely generate
# batches of augmented image data
train_generator = train_datagen.flow_from_directory(
        'train',  # this is the target directory
        target_size=(300, 300),  # all images will be resized to 300x300
        batch_size=batch_size,
        class_mode='binary')  # since we use binary_crossentropy loss, we need binary labels

# this is a similar generator, for validation data
validation_generator = test_datagen.flow_from_directory(
        'validation',
        target_size=(300, 300),
        batch_size=batch_size,
        class_mode='binary')

# TRAINING
tr_count1 = sum(len(files) for _, _, files in os.walk(r'C:\Users\Preeti\Codes\ImageApp\train\elephant'))
tr_count2 = sum(len(files) for _, _, files in os.walk(r'C:\Users\Preeti\Codes\ImageApp\train\tapir'))
tr_count = tr_count1+tr_count2
val_count1 = sum(len(files) for _, _, files in os.walk(r'C:\Users\Preeti\Codes\ImageApp\validation\elephant'))
val_count2 = sum(len(files) for _, _, files in os.walk(r'C:\Users\Preeti\Codes\ImageApp\validation\tapir'))
val_count = val_count1+val_count2

model.fit_generator(
        train_generator,
        steps_per_epoch=tr_count // batch_size,    # steps_per_epoch is iteration
        epochs=50,
          validation_data=validation_generator,
        validation_steps=val_count // batch_size)

model.save_weights('modelweight.h5')  # always save your weights after training or during training
print("Plot of the model")
model.save('modelsaved.h5')  # saving weights & structure
plot_model(model, to_file='model.png')
print("Model layers")
for i,layer in enumerate(model.layers):
  print(i,layer.name)

#Testing

class_dict = {0: 'Elephant', 1: 'Tapir'}
path_of_images = r"C:\Users\Preeti\Codes\ImageApp\test"
list_of_images = os.listdir(path_of_images)
testlist = []
for imagefile in list_of_images:
    plot_model(model, to_file='model.png')
    img = cv2.imread(os.path.join(path_of_images, imagefile))
    #Resize the image
    basewidth = 300
    hsize = 300
    img = Image.open(path_of_images +'\\'+imagefile)
    img = img.resize((basewidth,hsize), Image.ANTIALIAS)
    img.save(path_of_images +'\\'+imagefile) 
    x = img_to_array(img)
    x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 300, 300, 3)
    x.shape
    z = model.predict_classes(x)
    print(imagefile)
    print(class_dict[z[0][0]])
    model.predict_proba(x)
# file = open('etmodel.pkl', 'wb')
# pickle.dump(model, file)