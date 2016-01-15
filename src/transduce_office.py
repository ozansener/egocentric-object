import numpy as np
import alexnet_base as m_b # we will make this thing alexnet
import tensorflow as tf
import input_data # this will also change
import util_algebra as lalg
import util_logging as ul

class BasicTransduction(object):
    def __init__(self, source_folder, target_folder):
        self.logger = ul.DeepLogger()
        self.sess = tf.Session()
        self.BATCH_SIZE = 256
        self.NUM_EPOCHS = 5
        self.ALPHA = 500.0

        self.cur_batch=0

        self.source_dim = 1000 # alexnet FC is 1000 dimensional
        self.target_dim = 1000
        assert self.source_dim == self.target_dim

        self.source_labels = None  # GT labels of the source, one-hot
        self.target_labels = None  # computed labels of the target points
        self.target_labels_gt = None  # GT labels of the target, one-hot

        self.triple_images = tf.placeholder("float", shape=(None, 227*227*3*3)) #our new images are 227x227x3

        self.target_matrix = tf.placeholder("float", shape=(None, self.target_dim))  # features of the current batch
        self.source_matrix = tf.placeholder("float", shape=(None, self.source_dim))  # features of the full source
        self.source_matrix_batch = tf.placeholder("float", shape=(None, self.source_dim))  # features of the batch

        self.same_label = tf.placeholder("float", shape=(None, self.target_dim))
        self.diff_label = tf.placeholder("float", shape=(None, self.target_dim))

        self.diff_diff = tf.placeholder("float", shape=(None, self.target_dim))
        self.same_diff = tf.placeholder("float", shape=(None, self.target_dim))

        #self.w = tf.Variable(tf.diag(tf.constant(1.0, shape=[self.source_dim])))
        self.w = tf.get_variable('weight_matrix', 
                                 shape=[self.source_dim,self.source_dim], 
                                 initializer=tf.constant_initializer(np.eye(self.source_dim).flatten().tolist()))
        # We will define the model to further use
        self.w_half = tf.placeholder("float", shape=(self.source_dim, self.target_dim))

        self.im_ph, self.l_ph, self.kp_ph = m_b.placeholder_inputs()
        self.fc_out, self.softmax_out = m_b.inference(self.im_ph, self.kp_ph)

        # Read the data
        self.office_source = office_reader.read_data_set(source_folder)
        self.office_domain = office_reader.read_data_set(target_folder)

        self.source_features = np.zeros((self.office_source.num_examples,1000))
        self.target_features = np.zeros((self.office_target.num_examples,1000))
        #tf.placeholder("float", shape=(self.mnist_source.train.num_examples,1024))
        self.loss_minimize = self.get_lost_fnc()
        self.old_loss_minimize = self.get_old_lost_fnc()
        print 'Initial Variable List:'
        print [tt.name for tt in tf.trainable_variables()]

        # new loss function is
        # TODO: Make it only optimize for fully connecccted
        self.train_step = tf.train.AdagradOptimizer(0.005).minimize(self.loss_minimize) #, var_list=[self.w]) # no feaeture learning
        #self.train_step = tf.train.AdagradOptimizer(0.005).minimize(self.loss_minimize) #, var_list=[self.w])
 
        diff_norm = tf.reduce_sum(tf.mul(self.diff_diff, tf.transpose(tf.matmul(self.w, tf.transpose(self.diff_diff)))),
                          reduction_indices=1)
        same_norm = tf.reduce_sum(tf.mul(self.same_diff, tf.transpose(tf.matmul(self.w, tf.transpose(self.same_diff)))),
                          reduction_indices=1)
        self.min_d = tf.argmin(diff_norm, 0)
        self.min_s = tf.argmin(same_norm, 0)

        self.ass_a = self.w.assign(self.w_half)

        init_w = tf.initialize_variables([self.w])
	init = tf.initialize_all_variables()
        self.sess.run(init)

    def featurize_source_and_target(self):
        """
        This function featurize the entire dataset at the beginning of the computation
        :return:
        """
        process_size = 100
        # we can not fit the entire mnist into GPU, so will do 1000 images per clock
        for b_id in range(self.office_source.num_examples/process_size):
            im_b = self.office_source.images[b_id*process_size:b_id*process_size+process_size] # next_batch(1000) 
            lb = self.office_source.labels[b_id*process_size:b_id*process_size+process_size] # next_batch(1000) 
            source_feat = self.fc_out
            feats = self.sess.run(source_feat, feed_dict={self.im_ph:im_b, self.kp_ph:1.0})
            self.source_features[b_id*process_size:b_id*process_size+process_size, :] = feats

        for b_id in range(self.office_target.num_examples/process_size):
            im_b = self.office_target.images[b_id*process_size:b_id*process_size+process_size] # next_batch(1000) 
            lb = self.office_target.labels[b_id*process_size:b_id*process_size+process_size] # next_batch(1000) 
            source_feat = self.fc_out
            feats = self.sess.run(source_feat, feed_dict={self.im_ph:im_b, self.kp_ph:1.0})
            self.target_features[b_id*process_size:b_id*process_size+process_size, :] = feats

        _source_labels = self.office_source.labels
        self.target_labels = np.zeros((self.BATCH_SIZE))
        _target_labels_gt = self.office_target.labels
        self.source_labels = np.zeros((self.office_source.num_examples))
        self.target_labels_gt = np.zeros((self.office_source.num_examples))

        for target_id in range(self.office_source.num_examples):
            self.target_labels_gt[target_id] = np.nonzero(_target_labels_gt[target_id])[0][0]

        for source_id in range(self.office_source.num_examples):
            self.source_labels[source_id] = np.nonzero(_source_labels[source_id])[0][0]

    def set_labeling_eval_function(self):
        # Compute the distance matrix
        distances_ij_e = tf.matmul(self.source_matrix, tf.matmul(self.w, tf.transpose(self.target_matrix)))
        distances_ii_e = tf.matmul(tf.mul(self.source_matrix,
                                        tf.transpose(tf.matmul(self.w, tf.transpose(self.source_matrix)))),
                                 tf.ones([1024, self.office_target.num_examples]))
        distances_jj_e = tf.matmul(tf.mul(self.target_matrix,
                                        tf.transpose(tf.matmul(self.w, tf.transpose(self.target_matrix)))),
                                 tf.ones([1024, self.mnist_source.train.num_examples]))
        distances_e = distances_ii_e + tf.transpose(distances_jj_e) - 2*distances_ij_e
        self.min_distances_e = tf.argmin(distances_e, 0)

    def set_labeling_function(self):
        # Compute the distance matrix
        distances_ij = tf.matmul(self.source_matrix, tf.matmul(self.w, tf.transpose(self.target_matrix)))
        distances_ii = tf.matmul(tf.mul(self.source_matrix,
                                        tf.transpose(tf.matmul(self.w, tf.transpose(self.source_matrix)))),
                                 tf.ones([1024, self.BATCH_SIZE]))
        distances_jj = tf.matmul(tf.mul(self.target_matrix,
                                        tf.transpose(tf.matmul(self.w, tf.transpose(self.target_matrix)))),
                                 tf.ones([1024, self.mnist_source.train.num_examples]))
        distances = distances_ii + tf.transpose(distances_jj) - 2*distances_ij
        self.min_distances = tf.argmin(distances, 0)
        self.conf_scores = tf.reduce_min(distances, reduction_indices=[0])

    def label_target(self, is_second, batch_begin):
        # TODO: Make it use entire source set
        min_distances_eval, self.cur_scores = self.sess.run([self.min_distances, self.conf_scores], feed_dict=self.cur_data)
        if batch_begin%50 == 0:
            self.logger.write_nn(min_distances_eval, self.cur_batch, is_second)
        for target_id in range(self.BATCH_SIZE):
            self.target_labels[target_id] = self.source_labels[min_distances_eval[target_id]]



    def get_old_lost_fnc(self):
        self.lss = tf.add( tf.add( tf.neg(tf.reduce_sum(tf.mul(tf.matmul(self.diff_label,self.w), self.diff_label), 1)),
                                   tf.reduce_sum(tf.mul(tf.matmul(self.same_label,self.w), self.same_label), 1)),
                           self.ALPHA)
        self.loss = 600000.0*tf.reduce_sum(tf.nn.relu(self.lss)) #+ tf.reduce_sum(tf.pow(self.w, 2))
       
        return self.loss



    def get_lost_fnc(self):
        y_res = self.triple_loss() 
        feat_loss = self.feature_loss_fnc(y_res)
        self.loss = 600000.0*feat_loss 
        return self.loss

    def triple_loss(self):
        # Input images: [Anchor;Positive;Negative]
        # input_image 	= tf.reshape(self.triple_images, [-1,28*28*3,1])        
        imsize = 227
        imchannel = 3
        npix = imsize*imsize*imchannel
        anc_image = tf.reshape(self.triple_images[:,0:npix], [-1, imsize,imsize,imchannel])
        pos_image = tf.reshape(self.triple_images[:,npix:2*npix], [-1, imsize,imsize,imchannel])
        neg_image = tf.reshape(self.triple_images[:,2*npix:3*npix], [-1, imsize,imsize,imchannel])
 
        anc_f,y1 = m_b.inference_reuse(anc_image, self.kp_ph)
        pos_f,y1 = m_b.inference_reuse(pos_image, self.kp_ph)
        neg_f,y1 = m_b.inference_reuse(neg_image, self.kp_ph)



        triple_l = tf.add( tf.add( tf.neg(tf.reduce_sum(tf.mul(tf.matmul(neg_f-anc_f,self.w), neg_f-anc_f), 1)),
                           tf.reduce_sum(tf.mul(tf.matmul(pos_f-anc_f,self.w), pos_f-anc_f), 1)),
                           self.ALPHA)

        return tf.nn.relu(triple_l)

    def feature_loss_fnc(self, y_res):
        # This is same loss defined over input images to learn the features
        # input image is defined as [Anchor;Positive;Negative]
        return tf.reduce_sum(y_res)

    def learn_metric(self):
        #print self.cur_scores.shape
        #for target_id in range(self.BATCH_SIZE):
        #    print self.target_labels[target_id+self.cur_batch*self.BATCH_SIZE], np.nonzero(self.target_labels_gt[target_id+self.cur_batch*self.BATCH_SIZE])[0][0], self.cur_scores[target_id]

        id_c = np.array(range(self.BATCH_SIZE)).reshape((self.BATCH_SIZE,1))
        
        min_s_l = []
        min_d_l = []
        trip_list = []
        # we start with computing same and diff label matrices
        for src_pt in range(self.BATCH_SIZE):
            source_gt_label = self.source_labels[self.cur_batch*self.BATCH_SIZE+src_pt]
            source_value = self.source_features[self.cur_batch*self.BATCH_SIZE+src_pt]

            target_same = (source_gt_label ==
                             self.target_labels) # fix this
                             #self.target_labels_gt[self.cur_batch*self.BATCH_SIZE:(self.cur_batch+1)*self.BATCH_SIZE]) # fix this
            target_diff = np.logical_not(target_same)

            diff_diff_v = (self.cur_data[self.target_matrix][target_diff, :] - source_value)
            same_diff_v = (self.cur_data[self.target_matrix][target_same, :] - source_value)
            min_d_v, min_s_v = self.sess.run([self.min_d, self.min_s], feed_dict={self.diff_diff:diff_diff_v, self.same_diff:same_diff_v})
            min_s_l.append(same_diff_v[min_s_v])
            min_d_l.append(diff_diff_v[min_d_v])
         
            id_s = id_c[target_same, :]
            id_d = id_c[target_diff, :]

            # Here only push IDs instead
            anchor_im =  self.office_source.images[self.cur_batch*self.BATCH_SIZE+src_pt].reshape((1, 227*227*3))       
            pos_im = self.office_source.images[self.cur_batch*self.BATCH_SIZE+id_s[min_s_v]].reshape((1, 227*227*3))
            neg_im = self.office_source.images[self.cur_batch*self.BATCH_SIZE+id_d[min_d_v]].reshape((1, 227*227*3))
            dat = np.concatenate((anchor_im, pos_im, neg_im), axis=1)
            trip_list.append(dat)
        
     
        # send new images to the train_step
        self.sess.run(self.train_step, feed_dict={self.triple_images:np.vstack(trip_list)})
        # self.sess.run(self.train_step,
        #              feed_dict={self.diff_label:np.array(min_d_l), self.same_label:np.array(min_s_l)})
       

        W_old = self.sess.run(self.w, feed_dict={})
        W_new = lalg.project_to_psd(W_old)
        self.sess.run(self.ass_a, feed_dict={self.w_half: W_new})
        l_v2 = self.sess.run(self.old_loss_minimize, feed_dict={self.diff_label:np.array(min_d_l), self.same_label:np.array(min_s_l)})
        return l_v2

    def fill_batch(self):
        self.cur_batch += 1
        if (self.cur_batch+1)*self.BATCH_SIZE > self.office_source.num_examples:
            self.cur_batch = 0

        self.cur_data = {
            self.target_matrix: self.target_features[self.cur_batch*self.BATCH_SIZE:(self.cur_batch+1)*self.BATCH_SIZE, :],
            self.source_matrix: self.source_features,
            self.source_matrix_batch: self.source_features[
                                      self.cur_batch*self.BATCH_SIZE:(self.cur_batch+1)*self.BATCH_SIZE, :]
        }

    def evaluate_train_batch(self):
        tp = 0.0
        for bid in range(256):
            if self.target_labels[bid] == self.target_labels_gt[self.cur_batch*self.BATCH_SIZE+bid]:
                tp += 1.0
        return tp/256.0
 
    def evaluate_current(self):
        tp = 0.0
        for bid in range(self.office_target.num_examples/1000):
            min_distances_eval = self.sess.run(self.min_distances_e, feed_dict={self.source_matrix:self.source_features, self.target_matrix: self.target_features[bid*1000:(bid+1)*1000,:]})
 	    for idd in range(1000):
                if self.source_labels[min_distances_eval[idd]] == self.target_labels[1000*bid+idd]:
                    tp+=1.0
        return tp/self.mnist_target.test.num_examples
    
    def asym_eval(self):
        return tf.reduce_sum(tf.abs(tf.transpose(self.w) - self.w)), tf.reduce_sum(tf.pow(self.w, 2))


    def train_loop(self):
        # keep track of the batch size
        self.set_labeling_function()   
        self.set_labeling_eval_function()
        
        asy, nrm = self.asym_eval() 
        print 'Variables to learn are:' 
        print [tt.name for tt in tf.trainable_variables()]
        writer = tf.train.SummaryWriter("/tmp/dt_logs", self.sess.graph_def)
        for epoch_id in range(self.NUM_EPOCHS):
            for batch_begin in range(250):
                # fill the place holder with the batch
                self.fill_batch()
                # compute the target labels
                #acc = self.evaluate_current()
                #print "step {}, test accuracy {}".format(batch_begin+10000*epoch_id, acc)
                # solve the metric learning with lifted space
                self.label_target(False, batch_begin)
                if batch_begin % 50 == 0:
                    acc = self.evaluate_current()
                    self.logger.add_log("step {}, test accuracy before learning {}".format(batch_begin+250*epoch_id, acc))
                cur_l = self.learn_metric()
                if batch_begin % 50 == 0:
                    # Here we recompute features
                    self.featurize_source_and_target()
                    acc = self.evaluate_current()
                    self.logger.add_log("step {}, test accuracy after learning {}".format(batch_begin+250*epoch_id, acc))
                    self.label_target(True, batch_begin)

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

