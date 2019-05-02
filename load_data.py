# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 18:02:56 2017

@author: Eugene

Script contains function to load data
"""
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

import numpy as np
from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.contrib.learn.python.learn.datasets.mnist import DataSet
from tensorflow.python.framework import dtypes
import pickle
import os

MAX_INTENSITY = 255.0

def load_label_map(name, dataDir):
    """
    Load label map from file in directory dataDir
    name = {extracted, remaining}
    """
    # filenames
    file_label_map = os.path.join(dataDir, name + "_label_map.pkl")
    
    # load label map
    with open(file_label_map, 'rb') as f:
        label_map = pickle.load(f)

    return label_map    

def load_all_digits(dataDir, fraction_training):
    """
    Loads dataset with all digits from files in dataDir
    Return fraction_training examples from training datasets
    """
    # filenames
    file_all_train = os.path.join(dataDir, "all_train.npz")
    file_all_validation = os.path.join(dataDir, "all_validation.npz")
    file_all_test = os.path.join(dataDir, "all_test.npz")
    
    # all_digits data set
    # train data set
    all_train = np.load(file_all_train)
    all_train_labels = all_train['all_train_labels']
    all_train_images = all_train['all_train_images']
   
    # how many training examples to return
    num_train_to_extract = int(all_train_labels.shape[0] * fraction_training) 
   
    all_train = DataSet(all_train_images[:num_train_to_extract,:] * MAX_INTENSITY, all_train_labels[:num_train_to_extract,:], 
                                  dtype=dtypes.float32, reshape=False, one_hot=True)
    # validation data set
    all_validation = np.load(file_all_validation)
    all_validation_labels = all_validation['all_validation_labels']
    all_validation_images = all_validation['all_validation_images']
    all_validation = DataSet(all_validation_images * MAX_INTENSITY, all_validation_labels, 
                                  dtype=dtypes.float32, reshape=False, one_hot=True)
    # test data set
    all_test = np.load(file_all_test)
    all_test_labels = all_test['all_test_labels']
    all_test_images = all_test['all_test_images']
    all_test = DataSet(all_test_images * MAX_INTENSITY, all_test_labels, 
                                  dtype=dtypes.float32, reshape=False, one_hot=True)

    # combine all data sets
    all_digits = base.Datasets(train=all_train, validation=all_validation, test=all_test)
        
    return all_digits

    

def load_dataset(setName, name, dataDir, fraction_training):
    """
    Loads dataset with name 
    setName = {extracted, extracted_remapped, remaining, remaining_remapped} 
    from files in dataDir
    name - name of the dataset files
    Return fraction_training examples from training datasets
    """
    # files with data
    file_dataset_train = os.path.join(dataDir, setName + "_" + name + "_train.npz")
    file_dataset_validation = os.path.join(dataDir, setName + "_" + name + "_validation.npz")
    file_dataset_test = os.path.join(dataDir, setName + "_" + name + "_test.npz")
    
    # remapped data set
    # train data set
    d_train = np.load(file_dataset_train)
    dataset_train_labels = d_train[setName + '_train_labels']
    dataset_train_images = d_train[setName + '_train_images']
    
    # how many training examples to return
    num_train_to_extract = int(dataset_train_labels.shape[0] * fraction_training) 
    
    dataset_train = DataSet(dataset_train_images[:num_train_to_extract,:] * MAX_INTENSITY, dataset_train_labels[:num_train_to_extract,:], 
                                  dtype=dtypes.float32, reshape=False, one_hot=True)
    # validation data set
    d_validation = np.load(file_dataset_validation)
    dataset_validation_labels = d_validation[setName + '_validation_labels']
    dataset_validation_images = d_validation[setName + '_validation_images']
    dataset_validation = DataSet(dataset_validation_images * MAX_INTENSITY, dataset_validation_labels, 
                                  dtype=dtypes.float32, reshape=False, one_hot=True)

    # test data set
    d_test = np.load(file_dataset_test)
    dataset_test_labels = d_test[setName + '_test_labels']
    dataset_test_images = d_test[setName + '_test_images']
    dataset_test = DataSet(dataset_test_images * MAX_INTENSITY, dataset_test_labels, 
                                  dtype=dtypes.float32, reshape=False, one_hot=True)
    # combine all data sets
    dataset = base.Datasets(train=dataset_train, validation=dataset_validation, test=dataset_test)

    return dataset

def load_data(dataDir):
    """
    Loads all datasets from files in dataDir
    Return fraction_training examples from training datasets
    """

    # files with data
    file_extracted_train = os.path.join(dataDir, "extracted_train.npz")
    file_extracted_validation = os.path.join(dataDir, "extracted_validation.npz")
    file_extracted_test = os.path.join(dataDir, "extracted_test.npz")
    
    file_extracted_remapped_train = os.path.join(dataDir, "extracted_remapped_train.npz")
    file_extracted_remapped_validation = os.path.join(dataDir, "extracted_remapped_validation.npz")
    file_extracted_remapped_test = os.path.join(dataDir, "extracted_remapped_test.npz")

    file_remaining_train = os.path.join(dataDir, "remaining_train.npz")
    file_remaining_validation = os.path.join(dataDir, "remaining_validation.npz")
    file_remaining_test = os.path.join(dataDir, "remaining_test.npz")
    
    file_remaining_remapped_train = os.path.join(dataDir, "remaining_remapped_train.npz")
    file_remaining_remapped_validation = os.path.join(dataDir, "remaining_remapped_validation.npz")
    file_remaining_remapped_test = os.path.join(dataDir, "remaining_remapped_test.npz")
    
    file_all_train = os.path.join(dataDir, "all_train.npz")
    file_all_validation = os.path.join(dataDir, "all_validation.npz")
    file_all_test = os.path.join(dataDir, "all_test.npz")
    
    file_label_map = os.path.join(dataDir, "label_map.pkl")

    # load data and concatenate it into single dataset
    # extracted data set
    # train data set
    ext_train = np.load(file_extracted_train)
    extracted_train_labels = ext_train['extracted_train_labels']
    extracted_train_images = ext_train['extracted_train_images']
    
    extracted_train = DataSet(extracted_train_images* MAX_INTENSITY, extracted_train_labels, 
                                  dtype=dtypes.float32, reshape=False, one_hot=True)
    # validation data set
    ext_validation = np.load(file_extracted_validation)
    extracted_validation_labels = ext_validation['extracted_validation_labels']
    extracted_validation_images = ext_validation['extracted_validation_images']
    extracted_validation = DataSet(extracted_validation_images * MAX_INTENSITY, extracted_validation_labels, 
                                  dtype=dtypes.float32, reshape=False, one_hot=True)

    # test data set
    ext_test = np.load(file_extracted_test)
    extracted_test_labels = ext_test['extracted_test_labels']
    extracted_test_images = ext_test['extracted_test_images']
    extracted_test = DataSet(extracted_test_images * MAX_INTENSITY, extracted_test_labels, 
                                  dtype=dtypes.float32, reshape=False, one_hot=True)

    # combine all data sets
    extracted = base.Datasets(train=extracted_train, validation=extracted_validation, test=extracted_test)

    # extracted remapped data set
    # train data set
    ext_remap_train = np.load(file_extracted_remapped_train)
    extracted_remapped_train_labels = ext_remap_train['extracted_remapped_train_labels']
    extracted_remapped_train_images = ext_remap_train['extracted_remapped_train_images']
    
    extracted_remapped_train = DataSet(extracted_remapped_train_images* MAX_INTENSITY, extracted_remapped_train_labels, 
                                  dtype=dtypes.float32, reshape=False, one_hot=True)
    # validation data set
    ext_remap_validation = np.load(file_extracted_remapped_validation)
    extracted_remapped_validation_labels = ext_remap_validation['extracted_remapped_validation_labels']
    extracted_remapped_validation_images = ext_remap_validation['extracted_remapped_validation_images']
    extracted_remapped_validation = DataSet(extracted_remapped_validation_images * MAX_INTENSITY, extracted_remapped_validation_labels, 
                                  dtype=dtypes.float32, reshape=False, one_hot=True)

    # test data set
    ext_remap_test = np.load(file_extracted_remapped_test)
    extracted_remapped_test_labels = ext_remap_test['extracted_remapped_test_labels']
    extracted_remapped_test_images = ext_remap_test['extracted_remapped_test_images']
    extracted_remapped_test = DataSet(extracted_remapped_test_images * MAX_INTENSITY, extracted_remapped_test_labels, 
                                  dtype=dtypes.float32, reshape=False, one_hot=True)

    # combine all data sets
    extracted_remapped = base.Datasets(train=extracted_remapped_train, validation=extracted_remapped_validation, test=extracted_remapped_test)

    # remaining data set
    # train data set
    remaining_train = np.load(file_remaining_train)
    remaining_train_labels = remaining_train['remaining_train_labels']
    remaining_train_images = remaining_train['remaining_train_images']
    remaining_train = DataSet(remaining_train_images * MAX_INTENSITY, remaining_train_labels, 
                                  dtype=dtypes.float32, reshape=False, one_hot=True)
    # validation data set
    remaining_validation = np.load(file_remaining_validation)
    remaining_validation_labels = remaining_validation['remaining_validation_labels']
    remaining_validation_images = remaining_validation['remaining_validation_images']
    remaining_validation = DataSet(remaining_validation_images * MAX_INTENSITY, remaining_validation_labels, 
                                  dtype=dtypes.float32, reshape=False, one_hot=True)
    # test data set
    remaining_test = np.load(file_remaining_test)
    remaining_test_labels = remaining_test['remaining_test_labels']
    remaining_test_images = remaining_test['remaining_test_images']
    remaining_test = DataSet(remaining_test_images * MAX_INTENSITY, remaining_test_labels, 
                                  dtype=dtypes.float32, reshape=False, one_hot=True)

    # combine all data sets
    remaining = base.Datasets(train=remaining_train, validation=remaining_validation, test=remaining_test)
    
    # remaining remapped data set
    # train data set
    rem_remap_train = np.load(file_remaining_remapped_train)
    remaining_remapped_train_labels = rem_remap_train['remaining_remapped_train_labels']
    remaining_remapped_train_images = rem_remap_train['remaining_remapped_train_images']
    remaining_remapped_train = DataSet(remaining_remapped_train_images * MAX_INTENSITY, remaining_remapped_train_labels, 
                                  dtype=dtypes.float32, reshape=False, one_hot=True)
    # validation data set
    rem_remap_validation = np.load(file_remaining_remapped_validation)
    remaining_remapped_validation_labels = rem_remap_validation['remaining_remapped_validation_labels']
    remaining_remapped_validation_images = rem_remap_validation['remaining_remapped_validation_images']
    remaining_remapped_validation = DataSet(remaining_remapped_validation_images * MAX_INTENSITY, remaining_remapped_validation_labels, 
                                  dtype=dtypes.float32, reshape=False, one_hot=True)
    # test data set
    rem_remap_test = np.load(file_remaining_remapped_test)
    remaining_remapped_test_labels = rem_remap_test['remaining_remapped_test_labels']
    remaining_remapped_test_images = rem_remap_test['remaining_remapped_test_images']
    remaining_remapped_test = DataSet(remaining_remapped_test_images * MAX_INTENSITY, remaining_remapped_test_labels, 
                                  dtype=dtypes.float32, reshape=False, one_hot=True)

    # combine all data sets
    remaining_remapped = base.Datasets(train=remaining_train, validation=remaining_remapped_validation, test=remaining_remapped_test)
    
    # all_digits data set
    # train data set
    all_train = np.load(file_all_train)
    all_train_labels = all_train['all_train_labels']
    all_train_images = all_train['all_train_images']
    all_train = DataSet(all_train_images * MAX_INTENSITY, all_train_labels, 
                                  dtype=dtypes.float32, reshape=False, one_hot=True)
    # validation data set
    all_validation = np.load(file_all_validation)
    all_validation_labels = all_validation['all_validation_labels']
    all_validation_images = all_validation['all_validation_images']
    all_validation = DataSet(all_validation_images * MAX_INTENSITY, all_validation_labels, 
                                  dtype=dtypes.float32, reshape=False, one_hot=True)
    # test data set
    all_test = np.load(file_all_test)
    all_test_labels = all_test['all_test_labels']
    all_test_images = all_test['all_test_images']
    all_test = DataSet(all_test_images * MAX_INTENSITY, all_test_labels, 
                                  dtype=dtypes.float32, reshape=False, one_hot=True)

    # combine all data sets
    all_digits = base.Datasets(train=all_train, validation=all_validation, test=all_test)
        
    return (extracted, extracted_remapped, remaining, remaining_remapped, all_digits)

if __name__ == "__main__":
    dataDir = "C:\\Studying\\Data Mining 2\\Project\\dataminingproject\\data"
    
    fraction_training = 0.5
    
    
    #(label_map, remapped, remaining, all_digits) = load_data(dataDir)
    
    remaining_remapped = load_dataset("remaining_remapped", dataDir, fraction_training)
    remaining = load_dataset("remaining", dataDir, fraction_training)
    remaining_label_map = load_label_map("remaining", dataDir)
    
    
    
    #mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    
    np.set_printoptions(threshold=np.nan)
    #print(remapped.test.labels)
    #print(all_digits.test.images[0])
    print(remaining_remapped.train.labels.shape[0])
    print(remaining.train.labels.shape[0])
    
    #print(remaining.test.images[0])
    
    print(remaining_label_map)
    #print(np.argmax(mnist.test.labels, axis=1))
    
    sample = 3
    
    sample_label_remapped = remaining_remapped.train.labels[sample]
    sample_image_remapped = remaining_remapped.train.images[sample]
    
    sample_label_remaining = remaining.train.labels[sample]
    sample_image_remaining = remaining.train.images[sample]
    
    print("sample_label_remaining = ",sample_label_remaining)
    print("sample_label_remapped = ",sample_label_remapped)
    
    
    plt.imshow(sample_image_remaining.reshape((28, 28)))
    plt.gray()
    plt.figure()
    plt.imshow(sample_image_remapped.reshape((28, 28)))
    plt.gray()

#==============================================================================
#     sample_label_mnist = mnist.test.labels[sample]
#     sample_image_mnist = mnist.test.images[sample]
#     
#     print("sample_label_mnist = ",sample_label_mnist)
#     
#     plt.figure()
#     
#     plt.imshow(sample_image_mnist.reshape((28, 28)))
#     
#==============================================================================
    
    
    plt.show()