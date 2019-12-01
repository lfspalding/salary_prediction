import keras as k
import tensorflow as tf
import numpy as np
import random
import sklearn as sk
import csv
import pandas as pd
import gender_guesser.detector as gd
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.utils import model_to_dot
from keras.utils.np_utils import to_categorical
from keras import optimizers
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from numpy import array


def parse_data(input):
    with open(input) as csv_file:
        file_data = csv.reader(csv_file, delimiter=',')
        next(file_data)
        data = []
        job_id = {}
        job_index = 0
        for row in file_data:
            row[2].replace('"', '')
            if row[2] != 'JobTitle' or row[2] != 'Not provided':
                if row[2] not in job_id:
                    job_id[row[2]] = job_index
                    job_index += 1
                data.append([job_id[row[2]], int(float(row[7]))])
        #print(data)
        #print(job_id)
    return data

def get_gender(row, d):
    gender = 0  # 0 is male, 1 is female, 2 is undetermined
    name = row.split()
    gtemp = d.get_gender(name[0])
    if gtemp == 'male' or gtemp == 'mostly_male':
        gender = 0
    elif gtemp == 'female' or gtemp == 'mostly_female':
        gender = 1
    else: gender = 2
    return gender

def parse_gdata(input):
    d = gd.Detector(case_sensitive=False)
    with open(input) as csv_file:
        file_data = csv.reader(csv_file, delimiter=',')
        next(file_data)
        data = []
        job_id = {}
        job_index = 0
        for row in file_data:
            row[2].replace('"', '')
            if row[2] != 'JobTitle' or row[2] != 'Not provided':
                if row[2] not in job_id:
                    job_id[row[2]] = job_index
                    job_index += 1
                data.append([job_id[row[2]], get_gender(row[1], d), int(float(row[7]))])
        #print(data)
        #print(job_id)
    return data

# get data from csv
ngdata = parse_data('V1/Salaries.csv')
gdata = parse_gdata('V1/Salaries.csv')

# inputs and results
ngX = []
ngY = []
for value in ngdata:
    ngX.append(value[0])
    ngY.append(value[1])

gX = []
gY = []
for value in gdata:
    gX.append(value[0:2])
    gY.append(value[2])

#one hot encode x data
ngX = array(ngX)
print(ngX)
ngX_encoded = to_categorical(y=ngX, dtype='int16')
print(ngX_encoded.shape)

# separate test data
ngX_1, ngX_test, ngY_1, ngY_test = train_test_split(ngX_encoded, ngY, test_size=0.15)
gX_1, gX_test, gY_1, gY_test = train_test_split(gX, gY, test_size=0.15)

ngX_train, ngX_val, ngY_train, ngY_val = train_test_split(ngX_1, ngY_1, test_size=0.2)
gX_train, gX_val, gY_train, gY_val = train_test_split(gX_1, gY_1, test_size=0.2)

#create sequential model - ng
model = Sequential()
model.add(Dense(10, input_dim=ngX_encoded.shape[1], activation='sigmoid'))
#model.add(Dropout(rate=.2))
for i in range (3):
    model.add(Dense(50, activation='sigmoid'))
model.add(Dense(1, activation='linear'))
sgd = optimizers.SGD(lr = 0.001)
model.compile(optimizer=sgd,
              loss='mean_squared_error',
              metrics=['accuracy'])

batch_size = 15000
epochs = 10

#give training set to model
history = model.fit(x=ngX_train, y=ngY_train, batch_size=batch_size, epochs=epochs, verbose=2, validation_data=(ngX_val, ngY_val))
#evaluate = model.evaluate(x=test_data[0], y=test_data[1], batch_size=batch_size, )
predict = model.predict(ngX_test, batch_size=batch_size, verbose=1)
# cmatrix = confusion_matrix(test_labels.argmax(axis=1), predict.argmax(axis=1))
# print(cmatrix)

print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()



