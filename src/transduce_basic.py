import numpy as np
import mnist_base as m_b
import tensorflow as tf
import input_data

class BasicTransduction(object):
    def __init__(self, source_folder, target_folder):
        self.sess = tf.Session()
        self.BATCH_SIZE = 256
        self.NUM_EPOCHS = 2

        self.cur_batch=0

        self.source_dim = 1024
        self.target_dim = 1024
        assert self.source_dim == self.target_dim

        self.source_labels = None  # GT labels of the source, one-hot
        self.target_labels = None  # computed labels of the target points
        self.target_labels_gt = None  # GT labels of the target, one-hot

        self.target_matrix = tf.placeholder("float", shape=(None, self.target_dim))  # features of the current batch
        self.source_matrix = tf.placeholder("float", shape=(None, self.source_dim))  # features of the full source
        self.source_matrix_batch = tf.placeholder("float", shape=(None, self.source_dim))  # features of the batch

        self.same_label = tf.placeholder("float", shape=(None, self.target_dim))
        self.diff_label = tf.placeholder("float", shape=(None, self.target_dim))

        self.diff_diff = tf.placeholder("float", shape=(None, self.target_dim))
        self.same_diff = tf.placeholder("float", shape=(None, self.target_dim))

        self.w = tf.Variable(tf.diag(tf.constant(1.0, shape=[self.source_dim])))
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
        self.target_features = np.zeros((self.mnist_target.train.num_examples,1024))
        #tf.placeholder("float", shape=(self.mnist_source.train.num_examples,1024))
        self.loss_minimize = self.get_lost_fnc()
        self.train_step = tf.train.AdamOptimizer(1e-4).minimize(self.loss_minimize)

        diff_norm = tf.reduce_sum(tf.mul(self.diff_diff, tf.transpose(tf.matmul(self.w, tf.transpose(self.diff_diff)))),
                          reduction_indices=1)
        same_norm = tf.reduce_sum(tf.mul(self.same_diff, tf.transpose(tf.matmul(self.w, tf.transpose(self.same_diff)))),
                          reduction_indices=1)
        self.min_d = tf.argmin(diff_norm, 0)
        self.min_s = tf.argmin(same_norm)

    def featurize_source_and_target(self):
        """
        This function featurize the entire dataset at the beginning of the computation
        :return:
        """
        # we can not fit the entire mnist into GPU, so will do 1000 images per clock
        for b_id in range(self.mnist_source.train.num_examples/1000):
            im_b, lb = self.mnist_source.train.next_batch(1000) 
            source_feat = self.fc_out
            feats = self.sess.run(source_feat, feed_dict={self.im_ph:im_b, self.kp_ph:1.0})
            self.source_features[b_id*1000:b_id*1000+1000, :] = feats

        for b_id in range(self.mnist_target.train.num_examples/1000):
            im_b, lb = self.mnist_target.train.next_batch(1000) 
            source_feat = self.fc_out
            feats = self.sess.run(source_feat, feed_dict={self.im_ph:im_b, self.kp_ph:1.0})
            self.target_features[b_id*1000:b_id*1000+1000, :] = feats

        _source_labels = self.mnist_source.train.labels
        self.target_labels = np.zeros((self.BATCH_SIZE))
        _target_labels_gt = self.mnist_target.train.labels
        self.source_labels = np.zeros((self.mnist_source.train.num_examples))
        self.target_labels_gt = np.zeros((self.mnist_target.train.num_examples))

        for target_id in range(self.mnist_target.train.num_examples):
            self.target_labels_gt[target_id] = np.nonzero(_target_labels_gt[target_id])[0][0]

        for source_id in range(self.mnist_source.train.num_examples):
            self.source_labels[target_id] = np.nonzero(_source_labels[source_id])[0][0]

    def label_target(self):
        # Compute the distance matrix
        distances_ij = tf.matmul(self.source_matrix, tf.matmul(self.w, tf.transpose(self.target_matrix)))
        distances_ii = tf.matmul(tf.mul(self.source_matrix,
                                        tf.transpose(tf.matmul(self.w, tf.transpose(self.source_matrix)))),
                                 tf.ones([1024, self.BATCH_SIZE]))
        distances_jj = tf.matmul(tf.mul(self.target_matrix,
                                        tf.transpose(tf.matmul(self.w, tf.transpose(self.target_matrix)))),
                                 tf.ones([1024, self.mnist_source.train.num_examples]))
        distances = distances_ii + tf.transpose(distances_jj) - 2*distances_ij


        min_distances = tf.argmin(distances, 0)

        conf_scores = tf.reduce_min(distances, reduction_indices=[0])
        min_distances_eval, self.cur_scores = self.sess.run([min_distances, conf_scores], feed_dict=self.cur_data)
        for target_id in range(self.BATCH_SIZE):
            self.target_labels[target_id+self.cur_batch*self.BATCH_SIZE] = self.source_labels[
                min_distances_eval[target_id]]

    def get_lost_fnc(self):
        lss = tf.add(tf.add(tf.transpose(self.diff_label)*self.w*self.diff_label,
                            tf.neg(tf.transpose(self.diff_label)*self.w*self.diff_label)),
                     self.ALPHA)

        loss = tf.reduce_sum(tf.nn.relu(lss))
        return loss

    def learn_metric(self):
        #print self.cur_scores.shape
        #for target_id in range(self.BATCH_SIZE):
        #    print self.target_labels[target_id+self.cur_batch*self.BATCH_SIZE], np.nonzero(self.target_labels_gt[target_id+self.cur_batch*self.BATCH_SIZE])[0][0], self.cur_scores[target_id]

        min_s_l = []
        min_d_l = []
        # we start with computing same and diff label matrices
        same_lab = self.cur_data[self.t]

        for src_pt in range(self.BATCH_SIZE):
            source_gt_label = self.source_labels[self.cur_batch*self.BATCH_SIZE+src_pt]
            source_value = self.source_features[self.cur_batch*self.BATCH_SIZE+src_pt]

            target_same = (source_gt_label ==
                             self.target_labels_gt[self.cur_batch*self.BATCH_SIZE:(self.cur_batch+1)*self.BATCH_SIZE]) # fix this
            target_diff = np.logical_not(target_same)

            diff_diff_v = (self.cur_data[self.source_matrix_batch][target_diff, :] - source_value)
            same_diff_v = (self.cur_data[self.source_matrix_batch][target_same, :] - source_value)

            min_d_v, min_s_v = self.sess.run([self.min_d, self.min_s], feed_dict={self.diff_diff:diff_diff_v, self.same_diff:same_diff_v})
            min_s_l.append(same_diff_v[min_s_v])
            min_d_l.append(diff_diff_v[min_d_v])

        self.sess.run(self.train_step,
                      feed_dict={self.diff_label:np.array(min_d_l), self.same_label:np.array(min_s_l)})

    def fill_batch(self):
        self.cur_batch += 1
        if (self.cur_batch+1)*self.BATCH_SIZE > self.mnist_source.train.num_examples:
            self.cur_batch = 0

        self.cur_data = {
            self.target_matrix: self.target_features[self.cur_batch*self.BATCH_SIZE:(self.cur_batch+1)*self.BATCH_SIZE, :],
            self.source_matrix: self.source_features,
            self.source_matrix_batch: self.source_features[
                                      self.cur_batch*self.BATCH_SIZE:(self.cur_batch+1)*self.BATCH_SIZE, :]
        }

    def evaluate_current(self):
        raise NotImplementedError()

    def train_loop(self):
        # keep track of the batch size
        for epoch_id in range(self.NUM_EPOCHS):
            for batch_begin in range(0, self.target.shape[0], self.BATCH_SIZE):
                # fill the place holder with the batch
                self.fill_batch()
                # compute the target labels
                self.label_target()
                # solve the metric learning with lifted space
                self.learn_metric()

                if batch_begin % 100 == 99:
                    self.evaluate_current()

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
