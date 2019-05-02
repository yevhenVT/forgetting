# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 11:47:52 2017

@author: Eugene

Script compares performance of different Fisher multipliers
"""
import matplotlib.pyplot as plt
import os
import numpy as np

scriptDir = "C:\\Studying\\Data Mining 2\\Project\\dataminingproject\\" # directory with scripts
resultsDir = os.path.join(scriptDir, "results") # directory with results

dataset = "new_2hidden"

#file_1 = "FM0"
#file_2 = "FM10000000"
#file_3 = "FM10000000"

file_1 = "new_2hidden_replays_batch10_old1"
file_2 = "new_2hidden_replays_batch10_old3"
file_3 = "new_2hidden_replays_batch10_old9"



#file_accuracy_1 = os.path.join(resultsDir, file_1 + "_" + dataset + ".npz")
#file_accuracy_2 = os.path.join(resultsDir, file_2 + "_" + dataset + ".npz")
#file_accuracy_3 = os.path.join(resultsDir, file_3 + "_" + dataset + ".npz")

file_accuracy_1 = os.path.join(resultsDir, file_1 + ".npz")
file_accuracy_2 = os.path.join(resultsDir, file_2 + ".npz")
file_accuracy_3 = os.path.join(resultsDir, file_3 + ".npz")


def load_accuracy(filename):
    """
    Function load accuracy from file filename
    """
    
    accuracies = np.load(filename)
    
    time = accuracies['time']
    
    testing_accuracy_second = accuracies['testing_accuracy_second']

    testing_accuracy_first = accuracies['testing_accuracy_first']

    
    return (time, testing_accuracy_first, testing_accuracy_second)
            

    
(time_1, testing_accuracy_first_1, testing_accuracy_second_1) = load_accuracy(file_accuracy_1)

(time_2, testing_accuracy_first_2, testing_accuracy_second_2) = load_accuracy(file_accuracy_2)

(time_3, testing_accuracy_first_3, testing_accuracy_second_3) = load_accuracy(file_accuracy_3)


# plot results on validation dataset

f = plt.figure()


ax1 = f.add_subplot(131)

ax1.plot(time_1, testing_accuracy_second_1, label="dataset 2 ", color='g', linewidth=3)
ax1.plot(time_1, testing_accuracy_first_1, label="dataset 1 ", color='r', linewidth=3)
#ax.plot(time_1, testing_accuracy_all_1, label="tt_all_"+file_1, color='r', marker='v')
ax1.set_xlabel("time (# trial)")
ax1.set_ylabel("accuracy")

ax1.set_xlim([0,20000])
ax1.set_ylim([0,1.0])

ax1.set_title("1 old in 10")
ax1.legend()

ax2 = f.add_subplot(132)

ax2.plot(time_2, testing_accuracy_second_2, label="dataset 2 ", color='g', linewidth=3)
ax2.plot(time_2, testing_accuracy_first_2, label="dataset 1 ", color='r', linewidth=3)
#ax.plot(time_2, testing_accuracy_all_2, label="tt_all_"+file_2, color='r', marker='X')

#ax.plot(time_3, testing_accuracy_second_3, label="tt_1_"+file_3, color='g', marker='o')
#ax.plot(time_3, testing_accuracy_first_3, label="tt_2_"+file_3, color='b', marker='o')
#ax.plot(time_3, testing_accuracy_all_3, label="tt_all_"+file_3, color='r', marker='o')

ax2.set_xlabel("time (# trial)")
ax2.set_ylabel("accuracy")

ax2.set_xlim([0,20000])
ax2.set_ylim([0,1.0])

ax2.set_title("3 old in 10")
ax2.legend()

ax3 = f.add_subplot(133)

ax3.plot(time_3, testing_accuracy_second_3, label="dataset 2 ", color='g', linewidth=3)
ax3.plot(time_3, testing_accuracy_first_3, label="dataset 1 ", color='r', linewidth=3)
#ax.plot(time_2, testing_accuracy_all_2, label="tt_all_"+file_2, color='r', marker='X')

#ax.plot(time_3, testing_accuracy_second_3, label="tt_1_"+file_3, color='g', marker='o')
#ax.plot(time_3, testing_accuracy_first_3, label="tt_2_"+file_3, color='b', marker='o')
#ax.plot(time_3, testing_accuracy_all_3, label="tt_all_"+file_3, color='r', marker='o')

ax3.set_xlabel("time (# trial)")
ax3.set_ylabel("accuracy")

ax3.set_xlim([0,20000])
ax3.set_ylim([0,1.0])

ax3.set_title("9 old in 10")
ax3.legend()


#box = ax.get_position()
#ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

# Put a legend to the right of the current axis
#ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.show()      

