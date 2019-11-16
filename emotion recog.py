<<<<<<< Updated upstream
{"nbformat":4,"nbformat_minor":0,"metadata":{"colab":{"name":"emotion recog.py","provenance":[],"collapsed_sections":[]},"kernelspec":{"name":"python3","display_name":"Python 3"},"accelerator":"GPU"},"cells":[{"cell_type":"markdown","metadata":{"id":"N4XzlWiKTYZy","colab_type":"text"},"source":[""]},{"cell_type":"code","metadata":{"id":"hbVhpN62LOF_","colab_type":"code","colab":{}},"source":["import pandas as pd\n","import numpy as np\n","import warnings\n","warnings.filterwarnings(\"ignore\")\n","data = pd.read_csv('fer2013.csv')\n","\n","width, height = 48, 48\n","\n","datapoints = data['pixels'].tolist()\n","\n","#getting features for training\n","X = []\n","for xseq in datapoints:\n","\n","    xx = [int(xp) for xp in xseq.split(' ')]\n","    xx = np.asarray(xx).reshape(width, height)\n","    X.append(xx.astype('float32'))\n","\n","X = np.asarray(X)\n","X = np.expand_dims(X, -1)\n","\n","#getting labels for training\n","y = pd.get_dummies(data['emotion']).as_matrix()\n","\n","#storing them using numpy\n","np.save('features', X)\n","np.save('labels', y)\n","\n","print(\"Preprocessing Done\")\n","print(\"Number of Features: \"+str(len(X[0])))\n","print(\"Number of Labels: \"+ str(len(y[0])))\n","print(\"Number of examples in dataset:\"+str(len(X)))\n","print(\"X,y stored in features.npy and labels.npy respectively\")"],"execution_count":0,"outputs":[]},{"cell_type":"code","metadata":{"id":"kiMTH_bwYGuA","colab_type":"code","colab":{}},"source":["import sys, os\n","import pandas as pd\n","import numpy as np\n","from sklearn.model_selection import train_test_split\n","from keras.models import Sequential\n","from keras.layers import Dense, Dropout, Activation, Flatten\n","from keras.layers import Conv2D, MaxPooling2D, BatchNormalization\n","from keras.losses import categorical_crossentropy\n","from keras.optimizers import Adam\n","from keras.regularizers import l2\n","\n","num_features = 64\n","num_labels = 7\n","batch_size = 64\n","epochs = 100\n","width, height = 48, 48\n","\n","x = np.load('./features.npy')\n","y = np.load('./labels.npy')\n","\n","x -= np.mean(x, axis=0)\n","x /= np.std(x, axis=0)\n","\n","\n","# splitting into training, validation and testing data\n","X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)\n","X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.1, random_state=41)\n","\n","#saving the test samples to be used later\n","np.save('modelXtest', X_test)\n","np.save('modelytest', y_test)"],"execution_count":0,"outputs":[]},{"cell_type":"code","metadata":{"id":"JLLObjSbavxW","colab_type":"code","colab":{}},"source":["#desinging the CNN\n","model = Sequential()\n","\n","model.add(Conv2D(num_features, kernel_size=(3, 3), activation='relu', input_shape=(width, height, 1), data_format='channels_last', kernel_regularizer=l2(0.01)))\n","model.add(Conv2D(num_features, kernel_size=(3, 3), activation='relu', padding='same'))\n","model.add(BatchNormalization())\n","model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))\n","model.add(Dropout(0.5))\n","\n","model.add(Conv2D(2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))\n","model.add(BatchNormalization())\n","model.add(Conv2D(2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))\n","model.add(BatchNormalization())\n","model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))\n","model.add(Dropout(0.5))\n","\n","model.add(Conv2D(2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))\n","model.add(BatchNormalization())\n","model.add(Conv2D(2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))\n","model.add(BatchNormalization())\n","model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))\n","model.add(Dropout(0.5))\n","\n","model.add(Conv2D(2*2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))\n","model.add(BatchNormalization())\n","model.add(Conv2D(2*2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))\n","model.add(BatchNormalization())\n","model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))\n","model.add(Dropout(0.5))\n","\n","model.add(Flatten())\n","\n","model.add(Dense(2*2*2*num_features, activation='relu'))\n","model.add(Dropout(0.4))\n","model.add(Dense(2*2*num_features, activation='relu'))\n","model.add(Dropout(0.4))\n","model.add(Dense(2*num_features, activation='relu'))\n","model.add(Dropout(0.5))\n","\n","model.add(Dense(num_labels, activation='softmax'))\n","\n","model.summary()"],"execution_count":0,"outputs":[]},{"cell_type":"code","metadata":{"id":"bY1frbb1a0h6","colab_type":"code","colab":{}},"source":["#Compliling the model with adam optimixer and categorical crossentropy loss\n","model.compile(loss=categorical_crossentropy,\n","              optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7),\n","              metrics=['accuracy'])\n","\n","#training the model\n","model.fit(np.array(X_train), np.array(y_train),\n","          batch_size=batch_size,\n","          epochs=epochs,\n","          verbose=1,\n","          validation_data=(np.array(X_valid), np.array(y_valid)),\n","          shuffle=True)\n","\n","#saving the  model to be used later\n","fer_json = model.to_json()\n","with open(\"fer.json\", \"w\") as json_file:\n","    json_file.write(fer_json)\n","model.save_weights(\"fer.h5\")\n","print(\"Saved model to disk\")"],"execution_count":0,"outputs":[]},{"cell_type":"code","metadata":{"id":"KJt12O5mORNf","colab_type":"code","colab":{}},"source":["# load json and create model\n","from __future__ import division\n","from keras.models import Sequential\n","from keras.layers import Dense\n","from keras.models import model_from_json\n","import numpy\n","import os\n","import numpy as np\n","import cv2\n","from keras.models import model_from_json\n","from matplotlib import pyplot as plt\n","\n","#loading the model\n","json_file = open('fer.json', 'r')\n","# json_file = open('fer.json', 'r')\n","loaded_model_json = json_file.read()\n","json_file.close()\n","loaded_model = model_from_json(loaded_model_json)\n","# load weights into new model\n","loaded_model.load_weights(\"fer.h5\")\n","print(\"Loaded model from disk\")\n","\n","#setting image resizing parameters\n","WIDTH = 48\n","HEIGHT = 48\n","x=None\n","y=None\n","labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']\n","\n","#loading image\n","im = Image.open(root_path + \"surprise.jpg\")\n","full_size_image = cv2.imread(\"im\")\n","print(\"Image Loaded\")\n","im = np.asarray(im)\n","gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)\n","face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')\n","eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')\n","faces = face_cascade.detectMultiScale(gray, 1.3, 5)\n","\n","#detecting faces\n","for (x, y, w, h) in faces:\n","        roi_gray = gray[y:y + h, x:x + w]\n","        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)\n","        cv2.normalize(cropped_img, cropped_img, alpha=0, beta=1, norm_type=cv2.NORM_L2, dtype=cv2.CV_32F)\n","        cv2.rectangle(full_size_image, (x, y), (x + w, y + h), (0, 255, 0), 1)\n","        #predicting the emotion\n","        yhat= loaded_model.predict(cropped_img)\n","        cv2.putText(full_size_image, labels[int(np.argmax(yhat))], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1, cv2.LINE_AA)\n","        print(\"Emotion: \"+labels[int(np.argmax(yhat))])\n","\n","cv2.imshow('Emotion', full_size_image)\n","cv2.waitKey()"],"execution_count":0,"outputs":[]}]}
=======
# -*- coding: utf-8 -*-
"""emotion recog.py

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1_Vy-gGZr2RKojT_DpK1F9PnLQHalWz6R
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
data = pd.read_csv('fer2013.csv')

width, height = 48, 48

datapoints = data['pixels'].tolist()

#getting features for training
X = []
for xseq in datapoints:

    xx = [int(xp) for xp in xseq.split(' ')]
    xx = np.asarray(xx).reshape(width, height)
    X.append(xx.astype('float32'))

X = np.asarray(X)
X = np.expand_dims(X, -1)

#getting labels for training
y = pd.get_dummies(data['emotion']).as_matrix()

#storing them using numpy
np.save('features', X)
np.save('labels', y)

print("Preprocessing Done")
print("Number of Features: "+str(len(X[0])))
print("Number of Labels: "+ str(len(y[0])))
print("Number of examples in dataset:"+str(len(X)))
print("X,y stored in features.npy and labels.npy respectively")

import sys, os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.regularizers import l2

num_features = 64
num_labels = 7
batch_size = 64
epochs = 100
width, height = 48, 48

x = np.load('./features.npy')
y = np.load('./labels.npy')

x -= np.mean(x, axis=0)
x /= np.std(x, axis=0)


# splitting into training, validation and testing data
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.1, random_state=41)

#saving the test samples to be used later
np.save('modelXtest', X_test)
np.save('modelytest', y_test)

#desinging the CNN
model = Sequential()

model.add(Conv2D(num_features, kernel_size=(3, 3), activation='relu', input_shape=(width, height, 1), data_format='channels_last', kernel_regularizer=l2(0.01)))
model.add(Conv2D(num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(2*2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(2*2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.5))

model.add(Flatten())

model.add(Dense(2*2*2*num_features, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(2*2*num_features, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(2*num_features, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(num_labels, activation='softmax'))

model.summary()

#Compliling the model with adam optimixer and categorical crossentropy loss
model.compile(loss=categorical_crossentropy,
              optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7),
              metrics=['accuracy'])

#training the model
model.fit(np.array(X_train), np.array(y_train),
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(np.array(X_valid), np.array(y_valid)),
          shuffle=True)

#saving the  model to be used later
fer_json = model.to_json()
with open("fer.json", "w") as json_file:
    json_file.write(fer_json)
model.save_weights("fer.h5")
print("Saved model to disk")

# load json and create model
from __future__ import division
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import numpy
import os
import numpy as np
import cv2
from keras.models import model_from_json
from matplotlib import pyplot as plt

#loading the model
json_file = open('fer.json', 'r')
# json_file = open('fer.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("fer.h5")
print("Loaded model from disk")

#setting image resizing parameters
WIDTH = 48
HEIGHT = 48
x=None
y=None
labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

#loading image
im = Image.open(root_path + "surprise.jpg")
full_size_image = cv2.imread("im")
print("Image Loaded")
im = np.asarray(im)
gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

#detecting faces
for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
        cv2.normalize(cropped_img, cropped_img, alpha=0, beta=1, norm_type=cv2.NORM_L2, dtype=cv2.CV_32F)
        cv2.rectangle(full_size_image, (x, y), (x + w, y + h), (0, 255, 0), 1)
        #predicting the emotion
        yhat= loaded_model.predict(cropped_img)
        cv2.putText(full_size_image, labels[int(np.argmax(yhat))], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
        print("Emotion: "+labels[int(np.argmax(yhat))])

cv2.imshow('Emotion', full_size_image)
cv2.waitKey()
>>>>>>> Stashed changes
