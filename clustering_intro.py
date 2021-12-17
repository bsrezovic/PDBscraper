from __future__ import absolute_import, division, print_function, unicode_literals  #change how some base functionalities work
#supress warnings for gpu optimization libraries (need cuda enabled gpu for the stuff in the warnings to work actually)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#separate probability module

import tensorflow_probability as tfp

#an example is kmeans, which isnt well implemented in tensor (its ok you can writ it yourself)

#used when you have features but not labels

#hiddden markov models

#changes between states depending on hidden probabilities (we dont directly observe states, just observations: 
#                 example: 2 states; hot or cold day;   on a hot day we observe some range of temperatures with some prob distr (mean, sd), sme for cold day)
#                                            Each state (hot o rcold day) has some probability of being followed by a different/same state

#making a simle weather model

tfd = tfp.distributions  # making a shortcut for later on
initial_distribution = tfd.Categorical(probs=[0.9, 0.1])  # day 1 is 90% chance to be hot
transition_distribution = tfd.Categorical(probs=[[0.3, 0.7],
                                                 [0.2, 0.8]])  # refer to chances day transitions to other day or stay the same
observation_distribution = tfd.Normal(loc=[20., 35.], scale=[-5., 4.])  # temperturea are nirmally distributed

# the loc argument represents the mean and the scale is the standard devitation

#initialize the model with the variables
model = tfd.HiddenMarkovModel(
    initial_distribution=initial_distribution,
    transition_distribution=transition_distribution,
    observation_distribution=observation_distribution,
    num_steps=7)

#Goal: get  mean expected temepratures at every day

#to do that we create a partially defined computation; ergo a tensor

mean = model.mean()

#then we evaulute the whole session; all 7 days of moddeling by commanding:

with tf.compat.v1.Session() as sess:  
  print(mean.numpy())

#the result is always the same: its just the most probable temperature at every day