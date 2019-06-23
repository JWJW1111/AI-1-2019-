# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 19:08:11 2019

@author: JIWOO
"""
from PIL import Image
import os, glob, sys, numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import np_utils

# cat, dog 이미지 크롤링 ( google )
from google_images_download import google_images_download
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

def imageCrawling(keyword,dir):
    response = google_images_download.googleimagesdownload()
    
    arguments = {"keywords":keyword,
                 "limit":100,
                 "print_urls":True,
                 "no_directory":True,
                 "output_direction": dir
                 }
    paths = response.download(arguments)
    print(paths)

imageCrawling('cat','/Users/JI WOO/Desktop/project/cat')
imageCrawling('dog','/Users/JI WOO/Desktop/project/dog')

##

import time
import urllib.request
from selenium import webdriver
import json

path = 'C:/Users/JI WOO/Desktop/selenium/chromedriver.exe'
url_dog = "https://www.bing.com/images/search?q=dog%20image&qs=n&form=QBIR&qft=%20filterui%3Aphoto-photo%20filterui%3Alicense-L2_L3_L4_L5_L6_L7&sp=-1&pq=dog%20image&sc=2-9&sk=&cvid=1CD0DAA68AD9498C93180070D82A3788"
url_cat = "https://www.bing.com/images/search?q=cat%20image&qs=n&form=QBIR&qft=%20filterui%3Aphoto-photo%20filterui%3Alicense-L2_L3_L4_L5_L6_L7&sp=-1&pq=dog%20image&sc=2-9&sk=&cvid=1CD0DAA68AD9498C93180070D82A3788"

# dog 이미지 크롤링 in bing
browser = webdriver.Chrome(path)
browser.get(url_dog)

for i in range(7):
    browser.execute_script('window.scrollBy(0,10000)')
    time.sleep(3)

for idx, el in enumerate(browser.find_elements_by_class_name("mimg" or "ming rms_img" or "ming vimgld")):
    el.screenshot(str(idx)+".jpg")
    time.sleep(0.8)

browser.close()

# cat 이미지 크롤링 in bing
browser = webdriver.Chrome(path)
browser.get(url_cat)

for i in range(7):
    browser.execute_script('window.scrollBy(0,10000)')
    time.sleep(3)

for idx, el in enumerate(browser.find_elements_by_class_name("mimg" or "ming rms_img" or "ming vimgld")):
    el.screenshot(str(idx)+".jpg")
    time.sleep(0.8)

browser.close()

import os
# cat 이미지 이름 바꾸기 
input_path_1 = r'/Users/JI WOO/Desktop/project/dog vs cat/dataset/training_set/cats'
file_list_input_1 =os.listdir(input_path_1)
idx = 0
for i in file_list_input_1:
    idx +=1
    os.rename(r'/Users/JI WOO/Desktop/project/dog vs cat/dataset/training_set/cats/{0}'.format(i),r'/Users/JI WOO/Desktop/project/dog vs cat/dataset/training_set/cats/{0}.jpg'.format(idx))

# dog 이미지 이름 바꾸기 
input_path_1 = r'/Users/JI WOO/Desktop/project/dog vs cat/dataset/training_set/dogs'
file_list_input_1 =os.listdir(input_path_1)
idx = 0
for i in file_list_input_1:
    idx +=1
    os.rename(r'/Users/JI WOO/Desktop/project/dog vs cat/dataset/training_set/dogs/{0}'.format(i),r'/Users/JI WOO/Desktop/project/dog vs cat/dataset/training_set/dogs/{0}.jpg'.format(idx))

# train dataset = cats : 5,000 / dogs = 5,000

img_dir = './training_set'
img_dir
categories = ['cats','dogs']
np_classes = len(categories)    
np_classes

image_w = 64
image_h = 64   

pixel = image_h * image_w * 3    

X = []
y = []

for idx, cat in enumerate(categories):
    img_dir_detail = img_dir + "/" + cat
    files = glob.glob(img_dir_detail+"/*.jpg")

    for i, f in enumerate(files):
        try:
            img = Image.open(f)
            img = img.convert("RGB")
            img = img.resize((image_w, image_h))
            data = np.asarray(img)
            #Y는 0 아니면 1이니까 idx값으로 넣는다.
            X.append(data)
            y.append(idx)
            if i % 300 == 0:
                print(cat, " : ", f)
        except:
            print(cat, str(i)+" 번째에서 에러 ")
            
X = np.array(X)
Y = np.array(y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
X_train
xy = (X_train, X_test, Y_train, Y_test)
np.save(".dataset.npy", xy)

#######

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import keras.backend.tensorflow_backend as K

X_train, X_test, y_train, y_test = np.load('.dataset.npy')
print(X_train.shape)
print(X_test.shape)
print(X_train.shape[0])
print(np.bincount(y_train))
print(np.bincount(y_test))

image_w = 64
image_h = 64
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255

# model 만드는데 error가 발생하지 않도록
import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
print(tf.contrib.util.constant_value(tf.ones([1])))

with K.tf_ops.device('/device:GPU:0'):
    model = Sequential()
    model.add(Conv2D(32, (3,3), padding="same", input_shape=X_train.shape[1:], activation="relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(32, (3,3), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(64, (3,3), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3,3), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation="sigmoid"))
    
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    model_dir = './model'
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    model_path = model_dir + "/dog_cat_classify.model"
    
    checkpoint = ModelCheckpoint(filepath=model_path, monitor='val_loss', verbose=1, save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_loss', patience=7)

model.summary()


history = model.fit(X_train, y_train, batch_size=64, epochs=100, validation_split=0.15, callbacks=[checkpoint, early_stopping])

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['loss', 'val_loss', 'acc', 'val_acc'], loc='upper left')
plt.show()

###

from keras.models import load_model
import tensorflow as tf

seed = 5
tf.set_random_seed(seed)
np.random.seed(seed)

caltech_dir = './test_set'

image_w = 64
image_h = 64

pixels = image_h * image_w * 3

X = []
filenames = []
files = glob.glob(caltech_dir+"/*/*.*")
for i, f in enumerate(files):
    img = Image.open(f)
    img = img.convert("RGB")
    img = img.resize((image_w, image_h))
    data = np.asarray(img)

    filenames.append(f)
    X.append(data)

X = np.array(X)
X = X.astype(float) / 255
model = load_model('./model/dog_cat_classify.model')

prediction = model.predict(X)

np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
cnt = 0
for i in prediction:
    if i >= 0.5: print("해당 " + filenames[cnt].split("\\")[1] + filenames[cnt].split("\\")[2] + "  이미지는 개 로 추정됩니다.")
    else : print("해당 " + filenames[cnt].split("\\")[1] + filenames[cnt].split("\\")[2] + "  이미지는 고양이 로 추정됩니다.")
    cnt += 1