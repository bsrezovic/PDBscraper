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


CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']
# Lets define some constants to help us later on

train_path = tf.keras.utils.get_file(
    "iris_training.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv")
test_path = tf.keras.utils.get_file(
    "iris_test.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv")

train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)
# Here we use keras (a module inside of TensorFlow) to grab our datasets and read them into a pandas dataframe

train_y = train.pop('Species')
test_y = test.pop('Species')
print(train_y.shape)
def input_fn(features, labels, training=True, batch_size=256):
    # Convert the inputs to a Dataset.  this input function is simpler than the last one, no epochs specified, larger batch (bigger than sample?)
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle and repeat if you are in training mode.
    if training:
        dataset = dataset.shuffle(1000).repeat()
    
    return dataset.batch(batch_size)
# Feature columns describe how to use the input. Easy now, all are numeric

my_feature_columns = []
for key in train.keys():  # .keys returns columns
    my_feature_columns.append(tf.feature_column.numeric_column(key=key))
print(my_feature_columns)


#building a deep neural network for classification!
# Build a DNN with 2 hidden layers with 30 and 10 hidden nodes each.
classifier = tf.estimator.DNNClassifier(
    feature_columns=my_feature_columns,
    # Two hidden layers of 30 and 10 nodes respectively.
    hidden_units=[30, 10], 
    # The model must choose between 3 classes.
    n_classes=3)


#training
#lambda is a function defined in one line,
#remember in the first file input function returned a secodary embedded function -> this is similar, or an alternative
classifier.train(
    input_fn=lambda: input_fn(train, train_y, training=True), 
    steps=50) #steps are similar to epoch, instead of going through the data n epochs here we take 5000 steps total 
# We include a lambda to avoid creating an inner function previously

eval_result = classifier.evaluate(
    input_fn= lambda: input_fn(test, test_y, training=False)
)
print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))


#writting a simle function to predict a single flower when prompted

def input_fn(features, batch_size=256):
    # Convert the inputs to a Dataset without labels.
    return tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size)

features = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']
predict = {}

print("Please type numeric values as prompted.")
for feature in features:
  valid = True
  while valid: 
    val = input(feature + ": ")
    if not val.isdigit(): valid = False

  predict[feature] = [float(val)]   #the predict method only works on lists, even if we only have one value to predict from!

predictions = classifier.predict(input_fn=lambda: input_fn(predict))

print(predictions)

for pred_dict in predictions:     #predictions ends up being a bunch (a list?) of dictionaries
    class_id = pred_dict['class_ids'][0]
    probability = pred_dict['probabilities'][class_id]

    print('Prediction is "{}" ({:.1f}%)'.format(
        SPECIES[class_id], 100 * probability))
