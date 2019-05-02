# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 22:21:43 2017

@author: Eugene

class Network is a neural network consisting of user-defined number of
fully connected hidden layers
one of the class features is the ability to calculate Fisher information
matrix for keeping knowledge of previous classes
"""

import tensorflow as tf
import numpy as np
from functools import wraps

def lazy_property(function):
    attribute = '_' + function.__name__

    @property
    @wraps(function)
    def wrapper(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return wrapper

class Network(object):
    def __init__(self, num_neurons_in_layer, learning_rate=1e-3, Fisher_multiplier=0.0):
        """
        Neural network initialization. num_neurons_in_layer is a list
        with number of neurons in each layer
        """
        
        self.num_layers = len(num_neurons_in_layer) # total number of layers in the network
        self.num_neurons_in_layer = num_neurons_in_layer # number of neurons in each layer
        
        self.learning_rate = learning_rate # learning rate of opmitizer
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
            
        self.weights = [] # list with network weights
        self.biases = [] # list with biases
        
        self.grads_weights = [] # list of gradients for weights
        self.grads_biases = [] # list of gradients for biases
        
        
        
        self.Fisher_weights = [] # list of Fisher information matrices for weights
        self.Fisher_biases = [] # list of Fisher information matrices for biases
        
        self.learned_weights = [] # list of learned weights used for penalties with Fisher matrices
        self.learned_biases = [] # list of learned weights used for penalties with Fisher matrices
        
        # build the graph
        self.graph = tf.Graph()
        
        
        with self.graph.as_default():
            with tf.name_scope("Variables"):
                with tf.name_scope("Input"):
                    self.images = tf.placeholder(dtype=tf.float32, shape=[None, self.num_neurons_in_layer[0]],\
                                            name="input_images")
                    self.labels = tf.placeholder(dtype=tf.float32, shape=[None, self.num_neurons_in_layer[-1]],\
                                                 name="input_labels")
            
                with tf.name_scope("Network_weights_and_biases"):
                    # define all network layers
                    
                    for i in range(self.num_layers-1):
                        self.weights.append(weights_init([self.num_neurons_in_layer[i], \
                                                       self.num_neurons_in_layer[i+1]], \
                                                        name="W" + str(i) + str(i+1)))
                    
                        self.biases.append(bias_init([self.num_neurons_in_layer[i+1]], name="bias" + str(i+1)))
            
        
                with tf.name_scope("Fisher_matrices"):
                    # define all Fisher information matrices
                    for i in range(self.num_layers-1):
                        self.Fisher_weights.append(tf.Variable(tf.zeros(shape=[self.num_neurons_in_layer[i], \
                                                                  self.num_neurons_in_layer[i+1]], \
                                                                dtype=tf.float32), trainable=False))
                    
                        self.Fisher_biases.append(tf.Variable(tf.zeros(shape=[self.num_neurons_in_layer[i+1]], \
                                                                dtype=tf.float32), trainable=False))
                        
                        self.Fisher_multiplier = tf.constant(Fisher_multiplier, dtype=tf.float32)
                
                with tf.name_scope("Learned_weights"):
                    # define all learned weights used for penalties with Fisher
                    for i in range(self.num_layers-1):
                        self.learned_weights.append(tf.Variable(tf.zeros(shape=[self.num_neurons_in_layer[i], \
                                                                  self.num_neurons_in_layer[i+1]], \
                                                                dtype=tf.float32), trainable=False))
                    
                        self.learned_biases.append(tf.Variable(tf.zeros(shape=[self.num_neurons_in_layer[i+1]], \
                                                             dtype=tf.float32), trainable=False))
            
            self.raw_output
            self.loss
            self.accuracy
            self.optimize
            
            # define gradients for weights and biases
            self.grads_weights = [g for g,v in self.optimizer.compute_gradients(self.loss, self.weights)]
            self.grads_biases = [g for g,v in self.optimizer.compute_gradients(self.loss, self.biases)]
                
            # define update operations to calculate Fisher information matrices
            self.update_Fisher_weights = [v.assign_add(tf.multiply(self.grads_weights[i], self.grads_weights[i])) \
                                                                   for i,v in enumerate(self.Fisher_weights)]
            
            self.update_Fisher_biases = [v.assign_add(tf.multiply(self.grads_biases[i], self.grads_biases[i])) \
                                                                   for i,v in enumerate(self.Fisher_biases)]
            
            # operations to update learned weights in the end of the training
            self.update_learned_weights = [self.learned_weights[i].assign(w) for i,w in enumerate(self.weights)]
            self.update_learned_biases = [self.learned_biases[i].assign(b) for i,b in enumerate(self.biases)]
            
            
            self.saver = tf.train.Saver()
            init = tf.global_variables_initializer()
        
        # create session and initialize all variables
        self.sess = tf.Session(graph=self.graph)
        self.sess.run(init)
    
    @lazy_property
    def raw_output(self):
        """
        raw output of the final layer with response to shown input images
        """
        activations = [None] * (self.num_layers-2) # list with activations of neurons in each layer except the output
        
        activations[0] = tf.nn.relu(tf.matmul(self.images, self.weights[0]) + self.biases[0]) # activation of the first layer
        
        # calculate activations of all hidden layers
        for i in range(self.num_layers-3):
            activations[i+1] = tf.nn.relu(tf.matmul(activations[i], self.weights[i+1]) + self.biases[i+1])
        
        # raw output of the last layer
        output = tf.matmul(activations[-1], self.weights[-1]) + self.biases[-1]
        
        return output
    
    @lazy_property
    def loss(self):
        """
        Calculate the loss function based on true labels
        """
        # define penalties for weights going away from learned values
        weight_penalties = []
        bias_penalties = []
        
        for i in range(self.num_layers-1):
            weight_penalties.append(tf.reduce_sum(tf.multiply(self.Fisher_weights[i], tf.square(self.weights[i] - self.learned_weights[i]))))
            bias_penalties.append(tf.reduce_sum(tf.multiply(self.Fisher_biases[i], tf.square(self.biases[i] - self.learned_biases[i]))))
        
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.raw_output, labels=self.labels)) \
                        + self.Fisher_multiplier * (tf.add_n(weight_penalties) + tf.add_n(bias_penalties))
    
    @lazy_property
    def accuracy(self):
        """
        Operation to compute accuracy on the dataset
        """
        correct_prediction = tf.equal(tf.argmax(self.raw_output, 1), tf.argmax(self.labels, 1))
        return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    @lazy_property
    def optimize(self):
        """
        Do a step of gradient descent
        """
        return self.optimizer.minimize(self.loss)
    
    def calculate_accuracy(self, images, labels):
        """
        Estimate accuracy on dataset
        """
        return self.sess.run(self.accuracy, feed_dict={self.images:images, self.labels:labels})
    
    def train_on_batch(self, images, labels):
        """
        Do a training step on batch with images "images" and true labels "labels"
        """
        self.sess.run(self.optimize, feed_dict={self.images:images, self.labels:labels})
    
    def calculate_Fisher_information_matrices(self, images, labels):
        """
        Calculate Fisher information matrices on dataset
        """
        # loop through all samples in dataset
        for i in range(images.shape[0]):
            print("Sample: ",i)
            for j in range(self.num_layers - 1):
                self.sess.run(self.update_Fisher_weights[j], feed_dict={self.images: np.reshape(images[i], (1, -1)), \
                                                                        self.labels: np.reshape(labels[i], (1, -1))})
            
                self.sess.run(self.update_Fisher_biases[j], feed_dict={self.images: np.reshape(images[i], (1, -1)), \
                                                                        self.labels: np.reshape(labels[i], (1, -1))})
            
    
    def close_session(self):
        """
        Close tensorflow session
        """
        self.sess.close()
        
    def save_model(self, filename):
        """
        Saves model to the file "filename"
        """
        # update learned weights with the trained weights
        for i in range(self.num_layers-1):
            self.sess.run(self.update_learned_weights[i])
            self.sess.run(self.update_learned_biases[i])
               
        self.saver.save(self.sess, filename)
        
    def restore_model(self, filename):
        """
        Restores the model from the path filename
        """
        self.saver.restore(self.sess, filename)
        
def weights_init(shape, name):
    """
    Initialize weights for network layer
    """
    initial = tf.truncated_normal(shape, stddev=0.01)
    
    return tf.Variable(initial, name=name)

def bias_init(shape, name):
    """
    Initialize bias
    """
    initial = tf.constant(0.0, shape=shape)
    
    return tf.Variable(initial, name=name)
