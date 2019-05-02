Hi Ankur!

I slightly rewrote the class I use for the training.
Now I have fixed number of neurons in the output layer: 10.
Thus I don't need remapped labels for training and use "extracted" dataset
for the first training. I also gave names to datasets, so dataset with all digits
expect for 5 will be "extracted_012346789", where "012346789" is the dataset's name.

split_data downloads mnist datasets and splits them into extracted dataset,
containing desired digits and remaining dataset with all other digits.
Each dataset internally contains three datasets: training, validation and testing.
You can vary number of samples in each by selecting fractions of samples.
Each dataset has two variants: one with original labels and one is remapped labels.
I remapped the labels to 0 - num_digits.
For example, if your digits in the dataset are 1, 4, 7, then in the remapped
dataset they would have one-hot labels [1, 0, 0], [0, 1, 0] and [0, 0, 1]. 

First training is in trainNetwork.py script. Here you can choose original MNIST dataset
or extracted dataset for initial training. After training is completed, Fisher information matrix
is calculated for the training dataset.
Weights after the first training and Fisher matrices are saved with Saver to a file in directory "models".

Second training is done in trainNetwork_second.py script. I estimate performance on the
first and second datasets during second training and save results to the file in the directory "results".
You can vary the Fisher multiplier constant to constrain the weights differently.

You should first run split_data script to generate dataset. Then 
trainNetwork for the first training and trainNetwork_second 
for the second. You would need to 
change the name of directory variables in all scripts and you can play
with all the parameters you like.

Results can be compared with compareFisher.py script.
 

