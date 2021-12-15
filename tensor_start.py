from __future__ import absolute_import, division, print_function, unicode_literals  #change how some base functionalities work
#supress warnings for gpu optimization libraries (need cuda enabled gpu for the stuff in the warnings to work actually)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import tensorflow as tf
import matplotlib.pyplot as plt
#from six.moves import urllib
import numpy as np
import pandas as pd
import sklearn
from IPython.display import clear_output
#import tensorflow.compat.v2.feature_column as fc

test_tensor1 = tf.Variable(24,tf.int16)
test_tensor2 = tf.Variable([24,25],tf.int16)
test_tensor3 = tf.Variable([[24,25],[26,27],[28,29]],tf.int16)
print(test_tensor1)

#rank
print(tf.rank(test_tensor1))
print(tf.rank(test_tensor2))
print(tf.rank(test_tensor3))
#shape
print(test_tensor3.shape)

#Changing tensor shape, in most simple term flattening it to one dimension, more complx things later
tensor1 = tf.ones([1,2,3])  #a tensor with nothing but "1"s  in the shape specified [1,2,3] meaning [[[3 things][3 things]]]    / can also do zeros
print(tensor1)
tensor2 = tf.reshape(tensor1,[3,2,1]) # turn it into [    [ [x][x] ]  [   [x][x]   ]     [  [x][x] ]   ]    the first big bracket is implied, it represents the tensor, so it implies [1,3,2,1]
print(tensor2)
tensor3 = tf.reshape(tensor2,[2,-1])  # reshape to two lists and calculate therir size to fill in for "-1"
print(tensor3)
tensor4 = tf.reshape(tensor3,[-1])
print(tensor4)
#types of tensor: constant, variable, placeholder, sparsetensor  -- only variable is variable

#Evaluating tensors
#making sessions

#overview of core algorithms
#linear regression

# Load dataset.
dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv') # training data
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv') # testing data

#print(dftrain.head())
y_train = dftrain.pop('survived')  #pop out the columns from the dataset and assign them to variable
y_eval = dfeval.pop('survived')

#print(dftrain.loc[0:2])  #how subsetting works, the "normal syntax" (df[] can be used to find columns by name!)
print("------------------------------------------------")
#you can get some stats using .describe

#print(dftrain.describe())
#print(dftrain.shape)

#make some graphs using plt.show!
#dftrain.age.hist(bins = 20)

plot1 = plt.figure(1)  #have to initialise plots like this if you want multiple to be shown at once; one plt.show at the end will suffice!
dftrain.sex.value_counts().plot(kind='barh')  #frequency counts
plot2 = plt.figure(2)
dftrain["class"].value_counts().plot(kind='barh')  #has to be selected like this because class is a keyword

plot3 = plt.figure(3) #plotting percentage survival by sex for training data; we have t oconcatenate it now cuz we popped up there :(
pd.concat([dftrain, y_train], axis=1).groupby("sex").survived.mean().plot(kind = 'barh').set_xlabel('% survived')  
print(pd.concat([dftrain, y_train], axis=1).groupby("sex").survived.mean())


#making a model
print("____________________________")
#First step is spearating numerical from categorical columns
CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck',
                       'embark_town', 'alone']
NUMERIC_COLUMNS = ['age', 'fare']

#Then we must encode them into feture columns (male = 0, female = 1, etc.) to feed into the linear estimator
feature_columns = []
for feature_name in CATEGORICAL_COLUMNS:
  vocabulary = dftrain[feature_name].unique()  # gets a list of all unique values from given feature column
  feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))  #create a tensorflow feature column, provide vocab trough list
                                                                    #for lists of unique values the numeric value asigned will simply be the index, default value is -1 for unlisted 
for feature_name in NUMERIC_COLUMNS:
  feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))  #tf.float32 is also just the default value

#print(feature_columns)
#Model building

#the values are fed int the mdel in batches, because if we have to much data maybe we cant fit it into RAM all at once
# To do that we steal the input function from tensorflow website

def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
  def input_function():  # inner function, this will be returned
    ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))  # create tf.data.Dataset object with data and its label
    if shuffle:
      ds = ds.shuffle(1000)  # randomize order of data
    ds = ds.batch(batch_size).repeat(num_epochs)  # split dataset into batches of 32 and repeat process for number of epochs
    return ds  # return a batch of the dataset
  return input_function  # return a function object for use

train_input_fn = make_input_fn(dftrain, y_train)  # here we will call the input_function that was returned to us to get a dataset object we can feed to the model
eval_input_fn = make_input_fn(dfeval, y_eval, num_epochs=1, shuffle=False)   #epoch == 1 for training ofc


#linear regression
linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns)  # creates the estimator

linear_est.train(train_input_fn)  # input the training tf dataset into the estimator using the input function
result = linear_est.evaluate(eval_input_fn)  # get model metrics/stats by testing on testing data


print(result['accuracy'])  # the result variable is simply a dict of stats about our model
#getting the actual predictions from the model is easy
predictions = list(linear_est.predict(eval_input_fn))  #made it a list so we can loop through it easily
print(dfeval.loc[0])  #first person
print(predictions[0]["probabilities"][1]) #first persons survival probability


#plt.show()  #opens in a new window, have this on the end or commented out because it stops executon until you close the graph