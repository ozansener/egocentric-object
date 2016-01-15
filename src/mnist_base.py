import tensorflow.python.platform
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
    with tf.variable_scope('conv1') as scope:
        weights = tf.get_variable("weights", shape=[5,5,1,32],initializer=tf.truncated_normal_initializer(stddev = 0.1))        
        biases = tf.get_variable('biases', shape=[32],initializer=tf.constant_initializer(0.1))
 
        x_image = tf.reshape(input_image, [-1,28,28,1])        
        hidden_c1 = tf.nn.relu(conv2d(x_image, weights)+biases)
        hidden_pool1 = max_pool_2x2(hidden_c1)
        scope.reuse_variables()

    with tf.variable_scope('conv2') as scope:
        weights = tf.get_variable("weights", shape=[5,5,32,64],initializer=tf.truncated_normal_initializer(stddev = 0.1))        
        biases = tf.get_variable('biases', shape=[64],initializer=tf.constant_initializer(0.1))
       
        hidden_c2 = tf.nn.relu(conv2d(hidden_pool1, weights)+biases)
        hidden_pool2 = max_pool_2x2(hidden_c2)
        scope.reuse_variables()

    with tf.variable_scope('fully_connected') as scope:
        weights = tf.get_variable("weights", shape=[7*7*64, 1024],initializer=tf.truncated_normal_initializer(stddev = 0.1))        
        biases = tf.get_variable('biases', shape=[1024],initializer=tf.constant_initializer(0.1))
       
        hidden_pool2_flat = tf.reshape(hidden_pool2, [-1, 7*7*64])
        hidden_fc1 = tf.nn.relu(tf.matmul(hidden_pool2_flat, weights)+biases)
    
        # Add dropout
        hidden_fc1_drop = tf.nn.dropout(hidden_fc1, keep_prob)
        scope.reuse_variables()

    with tf.variable_scope('softmax') as scope:
        weights = tf.get_variable("weights", shape=[1024, 10],initializer=tf.truncated_normal_initializer(stddev = 0.1))        
        biases = tf.get_variable('biases', shape=[10],initializer=tf.constant_initializer(0.1))
        y_out = tf.nn.softmax(tf.matmul(hidden_fc1_drop, weights)+biases)
        scope.reuse_variables()

    return hidden_fc1, y_out


def inference_reuse(input_image, keep_prob):
    with tf.variable_scope('conv1', reuse=True) as scope:
        weights = tf.get_variable("weights", shape=[5,5,1,32],initializer=tf.truncated_normal_initializer(stddev = 0.1))        
        biases = tf.get_variable('biases', shape=[32],initializer=tf.constant_initializer(0.1))
 
        x_image = tf.reshape(input_image, [-1,28,28,1])        
        hidden_c1 = tf.nn.relu(conv2d(x_image, weights)+biases)
        hidden_pool1 = max_pool_2x2(hidden_c1)
        scope.reuse_variables()

    with tf.variable_scope('conv2', reuse=True) as scope:
        weights = tf.get_variable("weights", shape=[5,5,32,64],initializer=tf.truncated_normal_initializer(stddev = 0.1))        
        biases = tf.get_variable('biases', shape=[64],initializer=tf.constant_initializer(0.1))
       
        hidden_c2 = tf.nn.relu(conv2d(hidden_pool1, weights)+biases)
        hidden_pool2 = max_pool_2x2(hidden_c2)
        scope.reuse_variables()

    with tf.variable_scope('fully_connected', reuse=True) as scope:
        weights = tf.get_variable("weights", shape=[7*7*64, 1024],initializer=tf.truncated_normal_initializer(stddev = 0.1))        
        biases = tf.get_variable('biases', shape=[1024],initializer=tf.constant_initializer(0.1))
       
        hidden_pool2_flat = tf.reshape(hidden_pool2, [-1, 7*7*64])
        hidden_fc1 = tf.nn.relu(tf.matmul(hidden_pool2_flat, weights)+biases)
    
        # Add dropout
        hidden_fc1_drop = tf.nn.dropout(hidden_fc1, keep_prob)
        scope.reuse_variables()

    with tf.variable_scope('softmax', reuse=True) as scope:
        weights = tf.get_variable("weights", shape=[1024, 10],initializer=tf.truncated_normal_initializer(stddev = 0.1))        
        biases = tf.get_variable('biases', shape=[10],initializer=tf.constant_initializer(0.1))
        y_out = tf.nn.softmax(tf.matmul(hidden_fc1_drop, weights)+biases)
        scope.reuse_variables()

    return hidden_fc1, y_out

def loss(y_gt, y_out):
    return -tf.reduce_sum(y_gt*tf.log(y_out))

def acc(y_gt, y_out):
    correct_prediction = tf.equal(tf.argmax(y_out,1), tf.argmax(y_gt,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    return accuracy

