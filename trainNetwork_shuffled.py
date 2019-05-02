# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 23:49:27 2017

@author: Eugene

Script trains Network on shuffled MNIST data
using Fisher information matrix to keep previous 
knowledge
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

file_model = os.path.join(modelDir, "model_mnist_2hidden_64")
file_accuracy = os.path.join(resultsDir, "FM10_mnist_2hidden_3e-3.npz")


IMAGE_PIXELS = 28 * 28 # input image size (number of neurons in input layer)
NUM_HIDDEN_LAYER_NEURONS = 64 # number of neurons in hidden layer
NUM_LABELS = 10 # number of neurons in the output layer
learning_rate = 3e-3   # learning rate of optimizer
num_iterations = 60000 # number of training iterations
batch_size = 10 # mini-batch size
fraction_training = 1.0 # fraction of training datapoints to load
Fisher_multiplier = 10.0

num_neurons_in_layer = [IMAGE_PIXELS, NUM_HIDDEN_LAYER_NEURONS, NUM_HIDDEN_LAYER_NEURONS, NUM_LABELS] # num neurons in each layer

net = model.Network(num_neurons_in_layer, learning_rate, Fisher_multiplier)

# load data
first_dataset = load_data.load_all_digits(dataDir, 1.0)
second_dataset = load_data.load_all_digits(dataDir, 1.0)

np.random.seed(1991) # fix seed for random generator
perm = np.random.permutation(IMAGE_PIXELS)

print(perm)
# shuffle datasets

for i in range(len(second_dataset.train.images)):
    second_dataset.train.images[i] = second_dataset.train.images[i][perm]

for i in range(len(second_dataset.validation.images)):
    second_dataset.validation.images[i] = second_dataset.validation.images[i][perm]

for i in range(len(second_dataset.test.images)):
    second_dataset.test.images[i] = second_dataset.test.images[i][perm]



check_accuracy_frequency = 500 # how frequently we check accuracy on datasets

time = [] # time in trials

training_accuracy_second = [] # accuracy of training part of second dataset
validation_accuracy_second = [] # accuracy of validation part of second dataset
testing_accuracy_second = [] # accuracy of testing part of second dataset

testing_accuracy_first = [] # accuracy of testing part of first dataset

net.restore_model(file_model)
 
   
# repeat
for i in range(num_iterations):
    # get mini-batch
    batch = second_dataset.train.next_batch(batch_size)
    
    net.train_on_batch(batch[0], batch[1])
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
net.close_session()

np.savez(file_accuracy, time=time,
         testing_accuracy_second=testing_accuracy_second, \
         testing_accuracy_first=testing_accuracy_first)

del first_dataset
del second_dataset
