# -*- coding: utf-8 -*-
"""
Created on Wed May 24 07:42:19 2017

@author: Eugene

Script implements second training of neural network with replays
"""

import load_data
import tensorflow as tf
import model
import matplotlib.pyplot as plt
import os
import numpy as np

# global variables
# directory and file names
scriptDir = "C:\\Studying\\Data Mining 2\\Project\\dataminingproject\\" # directory with scripts
dataDir = os.path.join(scriptDir, "data") # directory with data sets
modelDir = os.path.join(scriptDir, "models") # directory with trained models
resultsDir = os.path.join(scriptDir, "results") # directory with results
file_model = os.path.join(modelDir, "model_new_2hidden_64")

file_accuracy = os.path.join(resultsDir, "new_2hidden_replays_batch10_old9.npz")


IMAGE_PIXELS = 28 * 28 # input image size (number of neurons in input layer)
NUM_HIDDEN_LAYER_NEURONS = 64 # number of neurons in hidden layer
NUM_LABELS = 10 # number of neurons in the output layer
learning_rate = 1e-3   # learning rate of optimizer
num_iterations = 20000 # number of training iterations
batch_size = 10 # mini-batch size
fraction_first_dataset = 0.9 # fraction of samples from first dataset
fraction_training = 1.0 # fraction of training datapoints to load
Fisher_multiplier = 0.0

batch_size_first = int(batch_size * fraction_first_dataset) # number of samples from first dataset
batch_size_second = batch_size - batch_size_first # number of samples from second dataset

num_neurons_in_layer = [IMAGE_PIXELS, NUM_HIDDEN_LAYER_NEURONS, NUM_HIDDEN_LAYER_NEURONS, NUM_LABELS] # num neurons in each layer

net = model.Network(num_neurons_in_layer, learning_rate, Fisher_multiplier)

# load data
first_dataset = load_data.load_dataset("extracted", "012346789", dataDir, 1.0)
second_dataset = load_data.load_dataset("remaining", "012346789", dataDir, 1.0)

check_accuracy_frequency = 75 # how frequently we check accuracy on datasets

time = [] # time in trials

training_accuracy_second = [] # accuracy of training part of second dataset
validation_accuracy_second = [] # accuracy of validation part of second dataset
testing_accuracy_second = [] # accuracy of testing part of second dataset

testing_accuracy_first = [] # accuracy of testing part of first dataset

net.restore_model(file_model)
   
# repeat
for i in range(num_iterations):
    # get mini-batch
    batch_first = first_dataset.train.next_batch(batch_size_first)
    batch_second = second_dataset.train.next_batch(batch_size_second)


    batch_images = np.concatenate((batch_first[0], batch_second[0]))
    batch_labels = np.concatenate((batch_first[1], batch_second[1]))

    
    net.train_on_batch(batch_images, batch_labels)
    #print("Weight: ",Network.sess.run(Network.W_fc1)[158])
    
    #print(Network.)
    # check model accuracy on datasets
    if i % check_accuracy_frequency == 0:
        print("Trial ",i)
        training_accuracy_second.append(net.calculate_accuracy(second_dataset.train.images, second_dataset.train.labels))
        validation_accuracy_second.append(net.calculate_accuracy(second_dataset.validation.images, second_dataset.validation.labels))
        testing_accuracy_second.append(net.calculate_accuracy(second_dataset.test.images, second_dataset.test.labels))
        
        testing_accuracy_first.append(net.calculate_accuracy(first_dataset.test.images, first_dataset.test.labels))
        
        time.append(i)
        

f = plt.figure()
ax = f.add_subplot(111)

ax.plot(time, training_accuracy_second, label="training_second")
ax.plot(time, validation_accuracy_second, label="validation_second")
ax.plot(time, testing_accuracy_second, label="test_second")

ax.plot(time, testing_accuracy_first, label="test_first")

ax.set_xlabel("time (# trial)")
ax.set_ylabel("accuracy")

plt.legend()
plt.show()      
    

#net.save_model(file_model)
np.savez(file_accuracy, time=time,
          testing_accuracy_second=testing_accuracy_second, \
          testing_accuracy_first=testing_accuracy_first)

net.close_session()

del first_dataset
del second_dataset


