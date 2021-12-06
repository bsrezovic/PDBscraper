import tensorflow as tf

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




