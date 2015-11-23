import input_data

import os.path
import time

import tensorflow.python.platform
import numpy
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import mnist_base as m_b

import click

@click.command()
@click.option('--source', prompt='Source folder', help='Name of the source folder.')
@click.option('--target', prompt='Target Folder', help='Name of the target folder.')
def train(source, target):
    mnist_source = input_data.read_data_sets(source, one_hot=True)
    mnist_target = input_data.read_data_sets(target, one_hot=True)

    with tf.Graph().as_default():
        im_ph, l_ph, kp_ph = m_b.placeholder_inputs()
        fc_out, softmax_out = m_b.inference(im_ph, kp_ph)
        lss = m_b.loss(l_ph, softmax_out)
 
        train_step = tf.train.AdamOptimizer(1e-4).minimize(lss)    

        accuracy = m_b.acc(l_ph, softmax_out)

        saver = tf.train.Saver()
        sess = tf.Session()
        init = tf.initialize_all_variables()
        sess.run(init)
        for i in range(20000):
            start_time = time.time()
            dict_data = m_b.fill_feed_dict(mnist_source, im_ph, l_ph, kp_ph, 50)
            sess.run(train_step,feed_dict=dict_data)
            if i%100 == 0:
                accuracy_v = sess.run(accuracy, feed_dict={im_ph:dict_data[im_ph],l_ph:dict_data[l_ph],kp_ph:1.0})
                print "step {}, training accuracy {}".format(i, accuracy_v)
            if i%1000 == 0: 
                accuracy_t = sess.run(accuracy, feed_dict={im_ph:mnist_target.test.images, l_ph:mnist_target.test.labels, kp_ph:1.0})
                print "test accuracy {}".format(accuracy_t)
	        saver.save(sess,'s_'+str(source)+'_t_'+str(target)+'_model',global_step=i)

if __name__ == '__main__':
    train()
