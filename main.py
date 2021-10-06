# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 14:45:06 2021

@author: anton
"""

print("This program creates an AI model that find out the value from a hand writing image of this number.")
print("This is called a MNIST model, because it is trained on the MNIST dataset.")

print("\nLoading modules...\n")
import os
from time import sleep
import random

import matplotlib.pyplot as plt
import numpy as np
from keras.utils import np_utils
from sklearn import datasets
from tensorflow.keras import Model, layers, optimizers, regularizers

os.environ['KMP_DUPLICATE_LIB_OK']='True'

print("Loading image datasets from sklearn")
digits = datasets.load_digits()

x = digits.images.reshape((len(digits.images), -1))
x.shape

y = np_utils.to_categorical(digits.target,10)

split_limit=1000
print(x.shape)
x_train = x[:split_limit]
y_train = y[:split_limit]
x_test = x[split_limit:]
y_test = y[split_limit:]

print("Display example images from training data")
for x in random.sample(range(1, 50), 3):
    img = x_test[x].reshape(8,8)
    print("Image of a "+str(np.argmax(y_test[x]))+" (close the window to continue)")
    plt.imshow(img)
    plt.show()

print("Configure and compile keras model")
img_input = layers.Input(shape=(64,))
tmp = layers.Dense(15,
                   activation='sigmoid')(img_input)
output = layers.Dense(10, 
                      activation='softmax')(tmp)

model = Model(img_input, output)
#model.summary()

l2_rate = 1e-4
for layer in model.layers:
    if hasattr(layer, 'kernel_regularizer'):
        layer.kernel_regularizer = regularizers.l2(l2_rate)
        layer.bias_regularizer = regularizers.l2(l2_rate)
        layer.activity_regularizer = regularizers.l2(l2_rate)

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(lr=0.1, momentum=0.9),
              metrics=['accuracy'])

print("Training the model with 50 epochs")
history = model.fit(x=x_train, y=y_train, validation_data=(x_test,y_test),
                    batch_size=100, epochs=50)

print("Training completed")

training_loss = history.history['loss']
test_loss = history.history['val_loss']

epoch_count = range(1, len(training_loss) + 1)

predictions = model.predict(x_test)


def plot_prediction(index):
    print("With this image of number "+str(np.argmax(y_test[index]))+" our model predicts (close the window to continue):")
    img = x_test[index].reshape(8,8)
    plt.imshow(img)
    plt.show()
    print("Number "+str(np.argmax(predictions[index])))
    if np.argmax(predictions[index]) == np.argmax(y_test[index]):
        print("It is True")
    else:
        print("It is False")

print("Checking predictions...")
for x in random.sample(range(1, 50), 5):
    plot_prediction(x)
    sleep(0.5)

print("Calculating accurary on 100 predictions")
rightpred=0
for index in random.sample(range(1, 200), 100):
    if np.argmax(predictions[index]) == np.argmax(y_test[index]):
        rightpred += 1
print("The accuracy on 100 predictions is "+str(rightpred)+"%")

if input("Do you want to see the history of the training loss? (y/n): ") == "y":
    plt.plot(epoch_count, training_loss, 'r--')
    plt.plot(epoch_count, test_loss, 'b-')
    plt.legend(['Training Loss', 'Test Loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()
