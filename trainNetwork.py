# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 23:49:27 2017

@author: Eugene

Script trains Network on MNIST data
"""
import load_data
import tensorflow as tf
import model
import matplotlib.pyplot as plt
import os

# global variables
# directory and file names
scriptDir = "C:\\Studying\\Data Mining 2\\Project\\dataminingproject\\" # directory with scripts
dataDir = os.path.join(scriptDir, "data") # directory with data sets
modelDir = os.path.join(scriptDir, "models") # directory with trained models
file_model = os.path.join(modelDir, "model_new_2hidden_64")

IMAGE_PIXELS = 28 * 28 # input image size (number of neurons in input layer)
NUM_HIDDEN_LAYER_NEURONS = 64 # number of neurons in hidden layer
NUM_LABELS = 10 # number of neurons in the output layer
learning_rate = 1e-3   # learning rate of optimizer
num_iterations = 60000 # number of training iterations
batch_size = 10 # mini-batch size
fraction_training = 1.0 # fraction of training datapoints to load

num_neurons_in_layer = [IMAGE_PIXELS, NUM_HIDDEN_LAYER_NEURONS, NUM_HIDDEN_LAYER_NEURONS, NUM_LABELS] # num neurons in each layer

net = model.Network(num_neurons_in_layer, learning_rate)


# load data
#dataset = load_data.load_all_digits(dataDir, 1.0)
dataset = load_data.load_dataset("extracted", "012346789", dataDir, 1.0)

check_accuracy_frequency = 1000 # how frequently we check accuracy on datasets

time = [] # time in trials
train_accuracy = [] # accuracy on training data set
validation_accuracy = [] # accuracy on validation data set
test_accuracy = [] # accuracy on test data set

#net.restore_model(file_model)
 
#print(net.calculate_accuracy(all_digits.train.images, all_digits.train.labels))
#print(net.calculate_accuracy(all_digits.validation.images, all_digits.validation.labels))
#print(net.calculate_accuracy(all_digits.test.images, all_digits.test.labels))
 
   
# repeat
for i in range(num_iterations):
    # get mini-batch
    batch = dataset.train.next_batch(batch_size)
    
    net.train_on_batch(batch[0], batch[1])
    #print("Weight: ",Network.sess.run(Network.W_fc1)[158])
    
    #print(Network.)
    # check model accuracy on datasets
    if i % check_accuracy_frequency == 0:
        print("Trial ",i)
        train_accuracy.append(net.calculate_accuracy(dataset.train.images, dataset.train.labels))
        validation_accuracy.append(net.calculate_accuracy(dataset.validation.images, dataset.validation.labels))
        test_accuracy.append(net.calculate_accuracy(dataset.test.images, dataset.test.labels))
        time.append(i)
        

f = plt.figure()
ax = f.add_subplot(111)

ax.plot(time, train_accuracy, label="training")
ax.plot(time, validation_accuracy, label="validation")
ax.plot(time, test_accuracy, label="test")

ax.set_xlabel("time (# trial)")
ax.set_ylabel("accuracy")

plt.legend()
plt.show()        

net.calculate_Fisher_information_matrices(dataset.train.images, dataset.train.labels)

net.save_model(file_model)
net.close_session()

del dataset