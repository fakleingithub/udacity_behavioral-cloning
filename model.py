import csv
from scipy import ndimage
import numpy as np

images = []
measurements = []

#read provided training data
lines = []
with open('data/driving_log.csv') as csvfile:
   reader = csv.reader(csvfile)
   for line in reader: 
      lines.append(line)

#pop headline in csv file
lines.pop(0)

for line in lines:
   for i in range(3):
      source_path = line[i]
      filename = source_path.split('/')[-1]
      current_path = 'data/IMG/' + filename
      image = ndimage.imread(current_path)
      images.append(image)
      #add steering correction for left and right camera images so car can handle sharp turns and off-mid situations
      correction = 0.2
      if i == 0:
         measurement = float(line[3])
      elif i == 1:
         measurment = float(line[3]) + correction
      elif i == 2: 
         measurement = float(line[3]) - correction
      measurements.append(measurement)

'''
lines = []
with open('fabianstraindata/standard/driving_log.csv') as csvfile:
   reader = csv.reader(csvfile)
   for line in reader: 
      lines.append(line)

for line in lines:
   source_path = line[0]
   filename = source_path.split('/')[-1]
   current_path = 'fabianstraindata/standard/IMG/' + filename
   image = ndimage.imread(current_path)
   images.append(image)
   measurement = float(line[3])
   measurements.append(measurement)
'''
#add counter-clockwise lap images to training data
lines = []
with open('fabianstraindata/standardcounterclockwise/driving_log.csv') as csvfile:
   reader = csv.reader(csvfile)
   for line in reader: 
      lines.append(line)

for line in lines:
   for i in range(3):
      source_path = line[i]
      filename = source_path.split('/')[-1]
      current_path = 'fabianstraindata/standardcounterclockwise/IMG/' + filename
      image = ndimage.imread(current_path)
      images.append(image)
      correction = 0.2
      if i == 0:
         measurement = float(line[3])
      elif i == 1:
         measurment = float(line[3]) + correction
      elif i == 2: 
         measurement = float(line[3]) - correction
      measurements.append(measurement)

# add images of jungle track to training data so model can generalize and avoid overfitting
lines = []
with open('fabianstraindata/jungle/driving_log.csv') as csvfile:
   reader = csv.reader(csvfile)
   for line in reader: 
      lines.append(line)

for line in lines:
   for i in range(3):
      source_path = line[i]
      filename = source_path.split('/')[-1]
      current_path = 'fabianstraindata/jungle/IMG/' + filename
      image = ndimage.imread(current_path)
      images.append(image)
      correction = 0.2
      if i == 0:
         measurement = float(line[3])
      elif i == 1:
         measurment = float(line[3]) + correction
      elif i == 2: 
         measurement = float(line[3]) - correction
      measurements.append(measurement)

X_train = np.array(images)
y_train = np.array(measurements)

from keras.models import Sequential 
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D

#model architecture of NVIDIA autonomous team End-to-End Deep Learning for Self-Driving Cars

model = Sequential()

#normalization
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))

#crop out top of image because the sky etc. is not relevant for training the model
model.add(Cropping2D(cropping=((70,25),(0,0))))

model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

#usage of adam optimizer, no manual learning rate parameterization needed
model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=2)

model.save('model.h5')
