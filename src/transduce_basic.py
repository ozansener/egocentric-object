import numpy as np
import mnist_base as m_b
import tensorflow as tf

class BasicTransduction(object):
    def __init__(self):
        self.target = None
        self.source = None
        self.source_labels = None
        self.target_labels = None
        self.feature_function = None
        self.sess = tf.Session()

        source_dim = 784
        target_dim = 784
        assert source_dim == target_dim
        self.target_matrix = tf.placeholder("float", shape=(None, target_dim))
        self.source_matrix = tf.placeholder("float", shape=(None, source_dim))
        self.w = tf.Variable(tf.diag(tf.constant(1, shape=[source_dim])))

        # We will define the model to further use
        self.im_ph, self.l_ph, self.kp_ph = m_b.placeholder_inputs()
        self.softmax_out = m_b.inference(self.im_ph, self.kp_ph)
        self.lss = m_b.loss(self.l_ph, self.softmax_out)
        self.accuracy = m_b.acc(self.l_ph, self.softmax_out)

    def featurize_source_and_target(self):
        raise NotImplementedError()

    def label_target(self):
        # Compute the distance matrix
        distances = tf.matmul(self.source_matrix, tf.matmul(self.w, tf.transpose(self.target_matrix)))
        min_distances = tf.argmax(distances, 0)
        for target_id in range(self.target.shape[0]):
            self.target_labels[target_id] = self.source_labels[min_distances[target_id]]

    def learn_metric(self):
        raise NotImplementedError()

    def fill_batch(self):
        raise NotImplementedError()

    def train_loop(self):
        # keep track of the batch size
        for epoch_id in range(self.NUM_EPOCHS):
            for batch_begin in range(0, self.target.shape[0], self.BATCH_SIZE):
                # fill the place holder with the batch
                self.fill_batch(batch_begin)
                # compute the target labels
                self.label_target()
                # solve the metric learning with lifted space
                self.learn_metric()

    def restore_the_model(self, file_name):
        num_var_in_graph = len(tf.all_variables())
        with tf.variable_scope('conv1', reuse=True):
            vc1w = tf.get_variable('weights') 
            vc1b = tf.get_variable('biases')
        with tf.variable_scope('conv2', reuse=True):
            vc2w = tf.get_variable('weights') 
            vc2b = tf.get_variable('biases')
        with tf.variable_scope('fully_connected', reuse=True):
            vfcw = tf.get_variable('weights') 
            vfcb = tf.get_variable('biases')
        with tf.variable_scope('softmax', reuse=True):
            vsw = tf.get_variable('weights') 
            vsb = tf.get_variable('biases')
        
        assert num_var_in_graph == len(tf.all_variables()) # Make sure no new variable is created     

        saver = tf.train.Saver({
            "conv1/weights":vc1w,
            "conv1/biases":vc1b,
            "conv2/weights":vc2w,
            "conv2/biases":vc2b,
            "fully_connected/weights":vfcw,
            "fully_connected/biases":vfcb,
            "softmax/weights":vsw,
            "softmax/biases":vsb
            })
        saver.restore(self.sess, file_name)
