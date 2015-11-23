import numpy as np
import mnist_base as m_b
import tensorflow as tf
import input_data

class BasicTransduction(object):
    def __init__(self, source_folder, target_folder):
        self.sess = tf.Session()
        self.BATCH_SIZE = 50
        self.cur_batch=0


        self.source_labels = None
        self.target_labels = None
        self.target_labels_gt = None        

        source_dim = 1024
        target_dim = 1024
        assert source_dim == target_dim
        self.target_matrix = tf.placeholder("float", shape=(None, target_dim))
        self.source_matrix = tf.placeholder("float", shape=(None, source_dim))
        self.w = tf.Variable(tf.diag(tf.constant(1.0, shape=[source_dim])))
        init_w = tf.initialize_variables([self.w])
        self.sess.run(init_w)
        # We will define the model to further use
        self.im_ph, self.l_ph, self.kp_ph = m_b.placeholder_inputs()
        self.fc_out, self.softmax_out = m_b.inference(self.im_ph, self.kp_ph)
        self.lss = m_b.loss(self.l_ph, self.softmax_out)
        self.accuracy = m_b.acc(self.l_ph, self.softmax_out)

        # Read the data
        self.mnist_source = input_data.read_data_sets(source_folder, one_hot=True)
        self.mnist_target = input_data.read_data_sets(target_folder, one_hot=True)

        self.source_features = np.zeros((self.mnist_source.train.num_examples,1024))
        self.target_features = np.zeros((self.mnist_source.train.num_examples,1024))
        #tf.placeholder("float", shape=(self.mnist_source.train.num_examples,1024))

    def featurize_source_and_target(self):
        # we can not fit the entire mnist into GPU, so will do 1000 images per clock
        for b_id in range(self.mnist_source.train.num_examples/1000):
            im_b, lb = self.mnist_source.train.next_batch(1000) 
            source_feat = self.fc_out
            feats = self.sess.run(source_feat, feed_dict={self.im_ph:im_b, self.kp_ph:1.0})
            self.source_features[b_id*1000:b_id*1000+1000,:]=feats

        for b_id in range(self.mnist_target.train.num_examples/1000):
            im_b, lb = self.mnist_target.train.next_batch(1000) 
            source_feat = self.fc_out
            feats = self.sess.run(source_feat, feed_dict={self.im_ph:im_b, self.kp_ph:1.0})
            self.target_features[b_id*1000:b_id*1000+1000,:]=feats

        self.source_labels = self.mnist_source.train.labels
        self.target_labels = np.zeros((self.mnist_target.train.num_examples))
        self.target_labels_gt = self.mnist_target.train.labels

    def label_target(self):
        # Compute the distance matrix
        distances = tf.matmul(self.source_matrix, tf.matmul(self.w, tf.transpose(self.target_matrix)))
        min_distances = tf.argmax(distances, 0)    
        conf_scores = tf.reduce_max(distances, reduction_indices=[0]) 
        min_distances_eval, self.cur_scores = self.sess.run([min_distances, conf_scores], feed_dict=self.cur_data)
        for target_id in range(self.BATCH_SIZE):
            self.target_labels[target_id+self.cur_batch*self.BATCH_SIZE] = np.nonzero(self.source_labels[min_distances_eval[target_id]])[0][0]

    def learn_metric(self):
        print self.cur_scores.shape
        for target_id in range(self.BATCH_SIZE):
            print self.target_labels[target_id+self.cur_batch*self.BATCH_SIZE], np.nonzero(self.target_labels_gt[target_id+self.cur_batch*self.BATCH_SIZE])[0][0], self.cur_scores[target_id]

    def fill_batch(self):
        self.cur_batch += 1
        self.cur_data = {self.target_matrix:self.target_features[self.cur_batch*self.BATCH_SIZE:(self.cur_batch+1)*self.BATCH_SIZE],
        self.source_matrix:self.source_features}

    def train_loop(self):
        # keep track of the batch size
        #for epoch_id in range(self.NUM_EPOCHS):
        #    for batch_begin in range(0, self.target.shape[0], self.BATCH_SIZE):
                # fill the place holder with the batch
        self.fill_batch(batch_begin)
                # compute the target labels
        self.label_target()
                # solve the metric learning with lifted space
        #self.learn_metric()

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
