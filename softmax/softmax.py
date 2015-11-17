# MNIST is a hand-written digits database. As for softmax, that will what we will do in multi-logistic regression to be able to sum the prob to be 1.

### Softmax Regression

import tensorflow as tf
import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot= True)

#The result is that mnist.train.images is a tensor (an n-dimensional array) with a shape of [60000, 784].The first dimension indexes the images and the second dimension indexes the pixels in each image.

#"one-hot vectors" equals to one-to-k encoding in ML course.

# mnist.train.images 60000*784 (28*28)
# mnist.train.labels 60000*10

#If you want to assign probabilities to an object being one of several different things, softmax is the thing to do. As usually in practice, we will put a layer of softmax at the final stage of NN.

# two steps:
#1\ calculate evidence ith: sum of weighted pixels tensity(the value in the mnist.train.images matrix and a bias term)
#2\ y = softmax(evidence) into predicted probabilities. here softmax serves as an activation function, shaping the evidence into a range we want.

x = tf.placeholder("float", [None, 784])

# when we try to run the computatino, we need to feed the needed info to the placeholder! here, none means that a dimension can be of any length!

# in Tensorflow, we use variable to handle additional inputs like weights and biases.

W = tf.Variable(tf.zeros([784, 10]))
# each column will be weights for each output units. 784*1
b = tf.Variable(tf.zeros([10]))

# it's all about Initialization, since we are about to learn W and b, it really doesn't matter what their initial vaule will be.

y = tf.nn.softmax(tf.matmul(x, W) + b)
# we use a trick here, since x is a 2D tensor with multiple inputs.

y_ = tf.placeholder("float", [None, 10])

cross_entropy = -tf.reduce_sum(y_*tf.log(y))

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
# Tensorflow here return us a single operation which when runned, will do a step of gradient descent training.

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# remember before we use placeholder to indicate that we need to feed some info when runned, here we feed the input data and label data into model! (get a "batch" of one hundred random data points from training set.)

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

print sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})












































