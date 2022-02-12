from __future__ import absolute_import, division, print_function, unicode_literals  #change how some base functionalities work
#supress warnings for gpu optimization libraries (need cuda enabled gpu for the stuff in the warnings to work actually)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}
import tensorflow as tf
from tensorflow import keras   #theses are the datasets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#creating our first official neural network

#import dataset
fashion_mnist = keras.datasets.fashion_mnist

#split the dataset
(train_images,train_labels),(test_images, test_labels) = fashion_mnist.load_data()
#this is a bunch of images and shit

print(train_images.shape)   #60000 images made , 10000for testing, pixel data for clothing

print(train_images[0,23,23]) #each pixel has 0-255 so theyre grayscale, these are numpy arrays 28

print(train_labels[:10]) #first 10 labels, theyre numbered because we have these classes:
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#show an image
plt.figure()
plt.imshow(train_images[1])
plt.colorbar()
plt.grid(False)
#plt.show()

#data preprocessing; we scale the values of the pixels to be between 0 and 1 
#by dividing them with 255
#this is because by default we tend to start the random biases and weights between -1 and 1
#so it helps if its all on the same scale

train_images = train_images/255.0
test_images = test_images/255.0


#bulding the architecture of the model


model = keras.Sequential([ #SEQUENTIAL == going left to right through network once! 
    keras.layers.Flatten(input_shape=(28, 28)),  # input layer (1), the pixels, 
    #flattened from an array to a vector
    keras.layers.Dense(700, activation='relu'),  # hidden layer (2)
    #layer 2 is dense meaning each neurn is conected to each one from  previous layer 
    #we chose 128 as number of neutrons here
    #activation function used is rectify linear unit
    keras.layers.Dense(10, activation='softmax') # output layer (3)
    #output is aslo obviously dense
    #uses softmax activation function, which gives probabilties!
    #all the 10 possible output neurons will add up to 0 or 1
    #highest is simply most probable, although is this reall probability? 
    #how does it depend on overlapp between classes?
])

#compiling the model by choosing optimizer etc

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


#training
model.fit(train_images, train_labels, epochs = 10)
#the accuracy printe hee reflects the accuracy on the training data, 
#actual testing has to be done by us like this:

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=1) 

print("Test loss", test_loss)
print('Test accuracy:', test_acc)

#predictions
predictions = model.predict(test_images)  #.predict wants an array even if you want just one
i = 0
for prediction in predictions[:3]:
    print(i)
    print(class_names[np.argmax(prediction)])
    print(class_names[test_labels[i]])
    i+=1
#fix this shit

