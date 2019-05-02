# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.

Script contains functions to split and save datasets.
"""
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.learn.python.learn.datasets.mnist import DataSet
from tensorflow.contrib.learn.python.learn.datasets.mnist import dense_to_one_hot
from tensorflow.python.framework import dtypes
from tensorflow.contrib.learn.python.learn.datasets import base
import matplotlib.pyplot as plt
import pickle
import os

num_classes = 10
MAX_INTENSITY = 255.0

def split_data_set(wanted_digits, fraction_train, fraction_test):
    """
    Splits MNIST dataset into two parts: one containing only digits in wanted_digits
    and second containing all remaining digits. Labels in dataset with wanted_digits
    are mapped to range 0:num_labels, where num_labels - number of wanted digits.
    Each of two datasets in turn is split into train, validation and test with number 
    of samples specified by fraction_train and fraction_test
    
    Output: (label_map, remapped, remaining)
    where label_map - dictionary containing mapping from original labels to new ones;
    remapped - dataset of type base.Datasets with wanted digits and remapped labels;
    remaining - dataset of type base.Datasets with remaining digits and original labels.
    """
    # read data set
    mnist = input_data.read_data_sets("MNIST_data/")

    # concatenate train, validation and test data
    concatenated_images = np.concatenate((mnist.test.images, mnist.validation.images, mnist.train.images),axis=0)
    concatenated_labels = np.concatenate((mnist.test.labels, mnist.validation.labels, mnist.train.labels),axis=0)

    # estimate class frequencies
    unique, counts = np.unique(concatenated_labels, return_counts=True)
    digit_frequency = dict(zip(unique, counts))
    print(digit_frequency)
   
    # rows to extract from dataset
    rows_to_extract = np.logical_or.reduce([concatenated_labels == x for x in wanted_digits])

    # extract samples corresponding to desired digits
    extracted_labels = concatenated_labels[rows_to_extract]
    extracted_images = concatenated_images[rows_to_extract]

    # remaining samples
    remaining_labels = concatenated_labels[np.logical_not(rows_to_extract)]
    remaining_images = concatenated_images[np.logical_not(rows_to_extract)]
   
    num_train_extracted = int(extracted_labels.shape[0] * fraction_train)
    num_test_extracted = int(extracted_labels.shape[0] * fraction_test)

    num_train_remaining = int(remaining_labels.shape[0] * fraction_train)
    num_test_remaining = int(remaining_labels.shape[0] * fraction_test)

    num_train_all = int(concatenated_labels.shape[0] * fraction_train)
    num_test_all = int(concatenated_labels.shape[0] * fraction_test)


    # map desired labels to range (0:num_labels)
    extracted_remapped_labels = np.empty(shape=extracted_labels.shape, dtype=np.int32)
    remaining_remapped_labels = np.empty(shape=remaining_labels.shape, dtype=np.int32)
    
    extracted_label_map = {} # dictionary containing label map
    remaining_label_map = {} #
                          
    # create map distionary for extracted digits
    for i in range(len(wanted_digits)):
        extracted_label_map[wanted_digits[i]] = i
    
    # create map distionary for remaining digits
    counter = 0
    for i in range(num_classes):
        if i not in wanted_digits:
            remaining_label_map[i] = counter
            counter += 1
        
            

    print("extracted_label map:" ,extracted_label_map)

    for i in range(extracted_labels.shape[0]):
        extracted_remapped_labels[i] = extracted_label_map[extracted_labels[i]]

    print("extracted_remapped_labels:" ,extracted_remapped_labels)

    print("remaining_label map:" ,remaining_label_map)

    for i in range(remaining_labels.shape[0]):
        remaining_remapped_labels[i] = remaining_label_map[remaining_labels[i]]

    print("remaining_remapped_labels:" ,remaining_remapped_labels)


    num_extracted_digits = len(wanted_digits) # number of classes in dataset with wanted_digits only

    # convert labels to one hot
    extracted_labels_one_hot = dense_to_one_hot(extracted_labels, num_classes)
    extracted_remapped_labels_one_hot = dense_to_one_hot(extracted_remapped_labels, num_extracted_digits)
    remaining_labels_one_hot = dense_to_one_hot(remaining_labels, num_classes)
    remaining_remapped_labels_one_hot = dense_to_one_hot(remaining_remapped_labels, num_classes - num_extracted_digits)
    
    concatenated_labels_one_hot = dense_to_one_hot(concatenated_labels, num_classes)
    
    print("extracted labels one hot: ",extracted_labels_one_hot)
    print("remaining labels one hot: ",remaining_labels_one_hot)
    
    print("extracted remapped labels one hot: ",extracted_remapped_labels_one_hot)
    print("remaining remapped labels one hot: ",remaining_remapped_labels_one_hot)
    
    # split datasets
    extracted_train = DataSet(extracted_images[:num_train_extracted] * MAX_INTENSITY, extracted_labels_one_hot[:num_train_extracted], 
                              dtype=dtypes.float32, reshape=False, one_hot=True)
    
    extracted_test = DataSet(extracted_images[num_train_extracted:(num_train_extracted + num_test_extracted)] * MAX_INTENSITY, 
                             extracted_labels_one_hot[num_train_extracted:(num_train_extracted + num_test_extracted)], 
                             dtype=dtypes.float32, reshape=False, one_hot=True)

    extracted_validation = DataSet(extracted_images[(num_train_extracted + num_test_extracted):] * MAX_INTENSITY, 
                             extracted_labels_one_hot[(num_train_extracted + num_test_extracted):], 
                             dtype=dtypes.float32, reshape=False, one_hot=True)

    
    extracted_remapped_train = DataSet(extracted_images[:num_train_extracted] * MAX_INTENSITY, extracted_remapped_labels_one_hot[:num_train_extracted], 
                              dtype=dtypes.float32, reshape=False, one_hot=True)

    extracted_remapped_test = DataSet(extracted_images[num_train_extracted:(num_train_extracted + num_test_extracted)] * MAX_INTENSITY, 
                             extracted_remapped_labels_one_hot[num_train_extracted:(num_train_extracted + num_test_extracted)], 
                             dtype=dtypes.float32, reshape=False, one_hot=True)

    extracted_remapped_validation = DataSet(extracted_images[(num_train_extracted + num_test_extracted):] * MAX_INTENSITY, 
                             extracted_remapped_labels_one_hot[(num_train_extracted + num_test_extracted):], 
                             dtype=dtypes.float32, reshape=False, one_hot=True)

    remaining_train = DataSet(remaining_images[:num_train_remaining] * MAX_INTENSITY, remaining_labels_one_hot[:num_train_remaining], 
                              dtype=dtypes.float32, reshape=False, one_hot=True)

    remaining_test = DataSet(remaining_images[num_train_remaining:(num_train_remaining + num_test_remaining)] * MAX_INTENSITY, 
                             remaining_labels_one_hot[num_train_remaining:(num_train_remaining + num_test_remaining)], 
                             dtype=dtypes.float32, reshape=False, one_hot=True)

    remaining_validation = DataSet(remaining_images[(num_train_remaining + num_test_remaining):] * MAX_INTENSITY, 
                             remaining_labels_one_hot[(num_train_remaining + num_test_remaining):], 
                             dtype=dtypes.float32, reshape=False, one_hot=True)
    
    remaining_remapped_train = DataSet(remaining_images[:num_train_remaining] * MAX_INTENSITY, remaining_remapped_labels_one_hot[:num_train_remaining], 
                              dtype=dtypes.float32, reshape=False, one_hot=True)

    remaining_remapped_test = DataSet(remaining_images[num_train_remaining:(num_train_remaining + num_test_remaining)] * MAX_INTENSITY, 
                             remaining_remapped_labels_one_hot[num_train_remaining:(num_train_remaining + num_test_remaining)], 
                             dtype=dtypes.float32, reshape=False, one_hot=True)

    remaining_remapped_validation = DataSet(remaining_images[(num_train_remaining + num_test_remaining):] * MAX_INTENSITY, 
                             remaining_remapped_labels_one_hot[(num_train_remaining + num_test_remaining):], 
                             dtype=dtypes.float32, reshape=False, one_hot=True)
    
    # data sets comprised of all digits
    all_train = DataSet(concatenated_images[:num_train_all] * MAX_INTENSITY, concatenated_labels_one_hot[:num_train_all], 
                              dtype=dtypes.float32, reshape=False, one_hot=True)

    all_test = DataSet(concatenated_images[num_train_all:(num_train_all + num_test_all)] * MAX_INTENSITY, concatenated_labels_one_hot[num_train_all:(num_train_all + num_test_all)], 
                              dtype=dtypes.float32, reshape=False, one_hot=True)
    
    all_validation = DataSet(concatenated_images[(num_train_all + num_test_all):] * MAX_INTENSITY, concatenated_labels_one_hot[(num_train_all + num_test_all):], 
                              dtype=dtypes.float32, reshape=False, one_hot=True)

    # combine data sets into single base.Datasets class
    extracted = base.Datasets(train=extracted_train, validation=extracted_validation, test=extracted_test)
    extracted_remapped = base.Datasets(train=extracted_remapped_train, validation=extracted_remapped_validation, test=extracted_remapped_test)
    remaining = base.Datasets(train=remaining_train, validation=remaining_validation, test=remaining_test)
    remaining_remapped = base.Datasets(train=remaining_remapped_train, validation=remaining_remapped_validation, test=remaining_remapped_test)
    
    all_digits = base.Datasets(train=all_train, validation=all_validation, test=all_test)
    
    return (extracted_label_map, remaining_label_map, extracted, extracted_remapped, \
            remaining, remaining_remapped, all_digits)
    
def save_data(dataDir, name, extracted_label_map, remaining_label_map, extracted, extracted_remapped, \
              remaining, remaining_remapped, all_digits):
    """
    Saves datasets to files in directory dataDir
    """
    # save datasets to files
    file_extracted_train = os.path.join(dataDir, "extracted_" + name + "_train.npz")
    file_extracted_validation = os.path.join(dataDir, "extracted_" + name + "_validation.npz")
    file_extracted_test = os.path.join(dataDir, "extracted_" + name + "_test.npz")
    
    file_extracted_remapped_train = os.path.join(dataDir, "extracted_remapped_" + name + "_train.npz")
    file_extracted_remapped_validation = os.path.join(dataDir, "extracted_remapped_" + name + "_validation.npz")
    file_extracted_remapped_test = os.path.join(dataDir, "extracted_remapped_" + name + "_test.npz")

    file_remaining_train = os.path.join(dataDir, "remaining_" + name + "_train.npz")
    file_remaining_validation = os.path.join(dataDir, "remaining_" + name + "_validation.npz")
    file_remaining_test = os.path.join(dataDir, "remaining_" + name + "_test.npz")
    
    file_remaining_remapped_train = os.path.join(dataDir, "remaining_remapped_" + name + "_train.npz")
    file_remaining_remapped_validation = os.path.join(dataDir, "remaining_remapped_" + name + "_validation.npz")
    file_remaining_remapped_test = os.path.join(dataDir, "remaining_remapped_" + name + "_test.npz")
    
    file_all_train = os.path.join(dataDir, "all_train.npz")
    file_all_validation = os.path.join(dataDir, "all_validation.npz")
    file_all_test = os.path.join(dataDir, "all_test.npz")

    file_extracted_label_map = os.path.join(dataDir, "extracted_" + name + "_label_map.pkl")
    file_remaining_label_map = os.path.join(dataDir, "remaining_" + name + "_label_map.pkl")

    # write train, validation and test datasets for dataset with extracted wanted digits
    np.savez(file_extracted_train, extracted_train_images=extracted.train.images, extracted_train_labels=extracted.train.labels)
    np.savez(file_extracted_validation, extracted_validation_images=extracted.validation.images, extracted_validation_labels=extracted.validation.labels)
    np.savez(file_extracted_test, extracted_test_images=extracted.test.images, extracted_test_labels=extracted.test.labels)

    # write train, validation and test datasets for dataset with extracted_remapped wanted digits
    np.savez(file_extracted_remapped_train, extracted_remapped_train_images=extracted_remapped.train.images, extracted_remapped_train_labels=extracted_remapped.train.labels)
    np.savez(file_extracted_remapped_validation, extracted_remapped_validation_images=extracted_remapped.validation.images, extracted_remapped_validation_labels=extracted_remapped.validation.labels)
    np.savez(file_extracted_remapped_test, extracted_remapped_test_images=extracted_remapped.test.images, extracted_remapped_test_labels=extracted_remapped.test.labels)

    # write train, validation and test datasets for dataset with remaining digits
    np.savez(file_remaining_train, remaining_train_images=remaining.train.images, remaining_train_labels=remaining.train.labels)
    np.savez(file_remaining_validation, remaining_validation_images=remaining.validation.images, remaining_validation_labels=remaining.validation.labels)
    np.savez(file_remaining_test, remaining_test_images=remaining.test.images, remaining_test_labels=remaining.test.labels)

    # write train, validation and test datasets for dataset with remaining remapped digits
    np.savez(file_remaining_remapped_train, remaining_remapped_train_images=remaining_remapped.train.images, remaining_remapped_train_labels=remaining_remapped.train.labels)
    np.savez(file_remaining_remapped_validation, remaining_remapped_validation_images=remaining_remapped.validation.images, remaining_remapped_validation_labels=remaining_remapped.validation.labels)
    np.savez(file_remaining_remapped_test, remaining_remapped_test_images=remaining_remapped.test.images, remaining_remapped_test_labels=remaining_remapped.test.labels)

    # write train, validation and test datasets for dataset with all digits
    np.savez(file_all_train, all_train_images=all_digits.train.images, all_train_labels=all_digits.train.labels)
    np.savez(file_all_validation, all_validation_images=all_digits.validation.images, all_validation_labels=all_digits.validation.labels)
    np.savez(file_all_test, all_test_images=all_digits.test.images, all_test_labels=all_digits.test.labels)


    with open(file_extracted_label_map, 'wb') as f:
        pickle.dump(extracted_label_map, f)

    with open(file_remaining_label_map, 'wb') as f:
        pickle.dump(remaining_label_map, f)


if __name__ == "__main__": 
    # desired digits to extract
    wanted_digits = [0, 1, 2, 3, 4, 6 ,7, 8, 9]
    name = "012346789"
    # split extracted and remaining samples into train, validation and test data sets
    fraction_train = 0.8 # fraction of training data
    fraction_test = 0.15 # fraction of test data
    
    dataDir = "C:\\Studying\\Data Mining 2\\Project\\dataminingproject\\data"
    
    (extracted_label_map, remaining_label_map, extracted, extracted_remapped, \
     remaining, remaining_remapped, all_digits) = split_data_set(wanted_digits, fraction_train, fraction_test)
    
    save_data(dataDir, name, extracted_label_map, remaining_label_map, extracted, \
              extracted_remapped, remaining, remaining_remapped, all_digits)
    
#==============================================================================
#     print("remapped train labels:", remapped.train.labels)
#     print("all_digits train labels:", all_digits.train.labels)
#     
#     batch = remapped.train.next_batch(20)
#     print(batch[0].shape)
#     print(batch[1])
#     
#==============================================================================
            
