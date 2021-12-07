from __future__ import absolute_import, division, print_function, unicode_literals  #change how some base functionalities work
#supress warnings for gpu optimization libraries (need cuda enabled gpu for the stuff in the warnings to work actually)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import tensorflow as tf
import matplotlib.pyplot as plt
from six.moves import urllib
import numpy as np
import pandas as pd
import sklearn
from IPython.display import clear_output
import tensorflow.compat.v2.feature_column as fc

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

print(dftrain.head())
y_train = dftrain.pop('survived')  #pop out the columns from the dataset and assign them to variable
y_eval = dfeval.pop('survived')

print(dftrain.loc[0:2])  #how subsetting works, the "normal syntax" (df[] can be used to find columns by name!)
print("------------------------------------------------")
#you can get some stats using .describe

print(dftrain.describe())
print(dftrain.shape)

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
# To do that we steal the input fucnction from tensorflow website







#plt.show()  #opens in a new window, have this on the end or commented out because it stops executon until you close the graph