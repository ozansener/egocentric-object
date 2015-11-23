import input_data

import os.path
import time

import tensorflow.python.platform
import numpy
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf



def placeholder_inputs():
    # this function returns the placeholders for training
    images_placeholder = tf.placeholder("float", shape=([None, 28*28]))
    labels_placeholder = tf.placeholder("float", shape=([None, 10]))
    keep_prob = tf.placeholder("float")

    return images_placeholder, labels_placeholder, keep_prob


def fill_feed_dict(data_set, images_pl, labels_pl, kp, batch_size):
    """Fills the feed_dict for training the given step.
  
    A feed_dict takes the form of:
    feed_dict = {
        <placeholder>: <tensor of values to be passed for placeholder>,
        ....
    }

    Args:
      data_set: The set of images and labels, from input_data.read_data_sets()
      images_pl: The images placeholder, from placeholder_inputs().
      labels_pl: The labels placeholder, from placeholder_inputs().

    Returns:
      feed_dict: The feed dictionary mapping from placeholders to values.
    """
    batch = data_set.train.next_batch(batch_size)
    feed_dict = {
        images_pl: batch[0],
        labels_pl: batch[1],
        kp: 0.5,
    }
    return feed_dict

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def inference(input_image, keep_prob):
    with tf.name_scope('conv1') as scope:
        weights = tf.Variable(tf.truncated_normal([5,5,1,32], stddev = 0.1), name='weights')
        biases = tf.Variable(tf.constant(0.1, shape=[32]), name='biases')
 
        x_image = tf.reshape(input_image, [-1,28,28,1])        
        hidden_c1 = tf.nn.relu(conv2d(x_image, weights)+biases)
        hidden_pool1 = max_pool_2x2(hidden_c1)

    with tf.name_scope('conv2') as scope:
        weights = tf.Variable(tf.truncated_normal([5,5,32,64], stddev = 0.1), name='weights')
        biases = tf.Variable(tf.constant(0.1, shape=[64]), name='biases')
        
        hidden_c2 = tf.nn.relu(conv2d(hidden_pool1, weights)+biases)
        hidden_pool2 = max_pool_2x2(hidden_c2)

    with tf.name_scope('fully_connected') as scope:
        weights = tf.Variable(tf.truncated_normal([7*7*64, 1024], stddev = 0.1), name='weights')
        biases = tf.Variable(tf.constant(0.1, shape=[1024]), name='biases')
        
        hidden_pool2_flat = tf.reshape(hidden_pool2, [-1, 7*7*64])
        hidden_fc1 = tf.nn.relu(tf.matmul(hidden_pool2_flat, weights)+biases)
    
        # Add dropout
        hidden_fc1_drop = tf.nn.dropout(hidden_fc1, keep_prob)

    with tf.name_scope('softmax') as scope:
        weights = tf.Variable(tf.truncated_normal([1024,10], stddev = 0.1), name='weights')
        biases = tf.Variable(tf.constant(0.1, shape=[10]), name='biases')
        y_out = tf.nn.softmax(tf.matmul(hidden_fc1_drop, weights)+biases)

    return y_out

def loss(y_gt, y_out):
    return -tf.reduce_sum(y_gt*tf.log(y_out))

def acc(y_gt, y_out):
    correct_prediction = tf.equal(tf.argmax(y_out,1), tf.argmax(y_gt,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    return accuracy

def train():
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    with tf.Graph().as_default():
        im_ph, l_ph, kp_ph = placeholder_inputs()
        softmax_out = inference(im_ph, kp_ph)
        lss = loss(l_ph, softmax_out)
 
        train_step = tf.train.AdamOptimizer(1e-4).minimize(lss)    

        accuracy = acc(l_ph, softmax_out)

        saver = tf.train.Saver()
        sess = tf.Session()
        init = tf.initialize_all_variables()
        sess.run(init)
        for i in range(20000):
            start_time = time.time()
            dict_data = fill_feed_dict(mnist, im_ph, l_ph, kp_ph, 50)
            sess.run(train_step,feed_dict=dict_data)
            if i%100 == 0:
                accuracy_v = sess.run(accuracy, feed_dict={im_ph:dict_data[im_ph],l_ph:dict_data[l_ph],kp_ph:1.0})
                print "step {}, training accuracy {}".format(i, accuracy_v)
            if i%1000 == 0: 
                accuracy_t = sess.run(accuracy, feed_dict={im_ph:mnist.test.images, l_ph:mnist.test.labels, kp_ph:1.0})
                print "test accuracy {}".format(accuracy_t)
	        saver.save(sess,'mnist-model',global_step=i)
def main(_):
    train()


if __name__ == '__main__':
  tf.app.run()
