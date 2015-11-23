import numpy as np
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
        saver = tf.train.Saver()
        saver.restore(self.sess, file_name)