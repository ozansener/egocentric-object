import numpy as np
import alexnet_base as m_b
import tensorflow as tf
import input_data
import util_algebra as lalg
import util_logging as ul

class BasicTransduction(object):
    def __init__(self, source_folder, target_folder):
        self.sess = tf.Session()
        self.im_ph, self.l_ph, self.kp_ph = m_b.placeholder_inputs()
        self.fc_out, self.softmax_out = m_b.inference(self.im_ph, self.kp_ph)
        #self.lss = m_b.loss(self.l_ph, self.softmax_out)
        #self.accuracy = m_b.acc(self.l_ph, self.softmax_out)

        print 'Initial Variable List:'
        print [tt.name for tt in tf.trainable_variables()]

	init = tf.initialize_all_variables()
        self.sess.run(init)

    def restore_the_model(self, file_name):
        data_dict = np.load(file_name).item()
        num_var_in_graph = len(tf.all_variables())
        with tf.variable_scope('conv1', reuse=True):
            vc1w = tf.get_variable('weights').assign(data_dict['conv1'][0]) 
            vc1b = tf.get_variable('biases').assign(data_dict['conv1'][1]) 
        with tf.variable_scope('conv2', reuse=True):
            vc2w = tf.get_variable('weights').assign(data_dict['conv2'][0]) 
            vc2b = tf.get_variable('biases').assign(data_dict['conv2'][1]) 
        with tf.variable_scope('conv3np', reuse=True):
            vc3w = tf.get_variable('weights').assign(data_dict['conv3'][0]) 
            vc3b = tf.get_variable('biases').assign(data_dict['conv3'][1]) 
        with tf.variable_scope('conv4np', reuse=True):
            vc4w = tf.get_variable('weights').assign(data_dict['conv4'][0]) 
            vc4b = tf.get_variable('biases').assign(data_dict['conv4'][1]) 
        with tf.variable_scope('conv5', reuse=True):
            vc5w = tf.get_variable('weights').assign(data_dict['conv5'][0]) 
            vc5b = tf.get_variable('biases').assign(data_dict['conv5'][1])
        with tf.variable_scope('fc6', reuse=True):
            vfc6w = tf.get_variable('weights').assign(data_dict['fc6'][0])
            vfc6b = tf.get_variable('biases').assign(data_dict['fc6'][1])
        with tf.variable_scope('fc7', reuse=True):
            vfc7w = tf.get_variable('weights').assign(data_dict['fc7'][0]) 
            vfc7b = tf.get_variable('biases').assign(data_dict['fc7'][1])
        with tf.variable_scope('softmax', reuse=True):
            vsw = tf.get_variable('weights').assign(data_dict['fc8'][0]) 
            vsb = tf.get_variable('biases').assign(data_dict['fc8'][1])
        self.sess.run([vc1w,vc1b,vc2w,vc2b,vc3w,vc3b,vc4w,vc4b,vc5w,vc5b,vfc6w,vfc7w,vfc7w,vfc7b,vsw,vsb])
        assert num_var_in_graph == len(tf.all_variables()) # Make sure no new variable is created     
        """
        saver = tf.train.Saver({
            "conv1/weights":vc1w,
            "conv1/biases":vc1b,
            "conv2/weights":vc2w,
            "conv2/biases":vc2b,
            "conv3np/weights":vc3w,
            "conv3np/biases":vc3b,
            "conv4np/weights":vc4w,
            "conv4np/biases":vc4b,
            "conv5/weights":vc5w,
            "conv5/biases":vc5b,
            "fc6/weights":vfc6w,
            "fc6/biases":vfc6b,
            "fc7/weights":vfc7w,
            "fc7/biases":vfc7b,
            "softmax/weights":vsw,
            "softmax/biases":vsb
            })
        saver.restore(self.sess, file_name)
        """
    def test(self):
        inp_im = np.random.random((10,227,227,3))
        fff = self.sess.run(self.fc_out, feed_dict={self.im_ph:inp_im, self.kp_ph:1.0})
        print fff.shape
bt = BasicTransduction(0, 0)
#saved_model =  np.load('alexnet.npy')
#[u'fc6', u'fc7', u'fc8', u'conv3', u'conv2', u'conv1', u'conv5', u'conv4']
#d_k =  saved_model.item()
#for kk in d_k.keys():
#    print kk, d_k[kk][0].shape, d_k[kk][1].shape
#print len(d_k['conv3'])
bt.restore_the_model("alexnet.npy")
bt.test()
