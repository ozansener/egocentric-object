import tensorflow.python.platform
import tensorflow as tf

def placeholder_inputs():
    # this function returns the placeholders for training
    images_placeholder = tf.placeholder("float", shape=([None, 227,227,3]))
    labels_placeholder = tf.placeholder("float", shape=([None, 1000]))
    keep_prob = tf.placeholder("float")

    return images_placeholder, labels_placeholder, keep_prob

def conv2d(x, W, p):
    return tf.nn.conv2d(x, W, strides=[1, 4, 4, 1], padding=p)

def conv2d1s(x, W, p):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding=p)


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 3, 3, 1],
                          strides=[1, 2, 2, 1], padding='VALID')


def lrn( inp, nam):
    return tf.nn.local_response_normalization(inp,
                                                  depth_radius=2,
                                                  alpha=2e-05,
                                                  beta=0.75,
                                                  bias=0.1,
                                                  name=nam)


def inference(input_image, keep_prob):
    param_to_send = []
    with tf.variable_scope('conv1') as scope:
        weights = tf.get_variable("weights", shape=[11,11,3,96],initializer=tf.truncated_normal_initializer(stddev = 0.1))        
        biases = tf.get_variable('biases', shape=[96],initializer=tf.constant_initializer(0.1))
        x_image = tf.reshape(input_image, [-1,227,227,3])        
        hidden_c1 = tf.nn.relu(conv2d(x_image, weights, 'VALID')+biases)
        hidden_pool1 = max_pool_2x2(lrn(hidden_c1, 'n1'))
        scope.reuse_variables()
    with tf.variable_scope('conv2') as scope:
        weights = tf.get_variable("weights", shape=[5,5,48,256],initializer=tf.truncated_normal_initializer(stddev = 0.1))        
        biases = tf.get_variable('biases', shape=[256],initializer=tf.constant_initializer(0.1))
       
        group = 2
        input_groups = tf.split(3, group, hidden_pool1)
        kernel_groups = tf.split(3, group, weights)
        output_groups = [conv2d1s(i, k, 'SAME') for i,k in zip(input_groups, kernel_groups)]
        conv = tf.concat(3, output_groups)     

        hidden_c2 = tf.nn.relu(conv+biases)
        hidden_pool2 = max_pool_2x2(lrn(hidden_c2, 'n2'))
        scope.reuse_variables()

    with tf.variable_scope('conv3np') as scope:
        weights = tf.get_variable("weights", shape=[3,3,256,384],initializer=tf.truncated_normal_initializer(stddev = 0.1))        
        biases = tf.get_variable('biases', shape=[384],initializer=tf.constant_initializer(0.1))
       
        hidden_c3 = tf.nn.relu(conv2d1s(hidden_pool2, weights, 'SAME')+biases)
        scope.reuse_variables()

    with tf.variable_scope('conv4np') as scope:
        weights = tf.get_variable("weights", shape=[3,3,192,384],initializer=tf.truncated_normal_initializer(stddev = 0.1))        
        biases = tf.get_variable('biases', shape=[384],initializer=tf.constant_initializer(0.1))

        group = 2
        input_groups = tf.split(3, group, hidden_c3)
        kernel_groups = tf.split(3, group, weights)
        output_groups = [conv2d1s(i, k, 'SAME') for i,k in zip(input_groups, kernel_groups)]
        conv = tf.concat(3, output_groups)     
      
        hidden_c4 = tf.nn.relu(conv+biases)
        scope.reuse_variables()
    with tf.variable_scope('conv5') as scope:
        weights = tf.get_variable("weights", shape=[3,3,192,256],initializer=tf.truncated_normal_initializer(stddev = 0.1))        
        biases = tf.get_variable('biases', shape=[256],initializer=tf.constant_initializer(0.1))

        group = 2
        input_groups = tf.split(3, group, hidden_c4)
        kernel_groups = tf.split(3, group, weights)
        output_groups = [conv2d1s(i, k, 'SAME') for i,k in zip(input_groups, kernel_groups)]
        conv = tf.concat(3, output_groups)      
       
        hidden_c5 = tf.nn.relu(conv+biases)
        hidden_pool5 = max_pool_2x2(hidden_c5)
        scope.reuse_variables()

    with tf.variable_scope('fc6') as scope:
        weights = tf.get_variable("weights", shape=[6*6*256, 4096],initializer=tf.truncated_normal_initializer(stddev = 0.1))        
        biases = tf.get_variable('biases', shape=[4096],initializer=tf.constant_initializer(0.1))
        hidden_pool5_flat = tf.reshape(hidden_pool5, [-1, 6*6*256])
        hidden_fc6 = tf.nn.relu(tf.matmul(hidden_pool5_flat, weights)+biases)
    
        # Add dropout
        scope.reuse_variables()

    with tf.variable_scope('fc7') as scope:
        weights = tf.get_variable("weights", shape=[4096, 4096],initializer=tf.truncated_normal_initializer(stddev = 0.1))        
        biases = tf.get_variable('biases', shape=[4096],initializer=tf.constant_initializer(0.1))
       
        hidden_fc7_flat = tf.reshape(hidden_fc6, [-1, 4096])
        hidden_fc7 = tf.nn.relu(tf.matmul(hidden_fc7_flat, weights)+biases)
    
        # Add dropout
        scope.reuse_variables()

    with tf.variable_scope('softmax') as scope:
        weights = tf.get_variable("weights", shape=[4096, 1000],initializer=tf.truncated_normal_initializer(stddev = 0.1))        
        biases = tf.get_variable('biases', shape=[1000],initializer=tf.constant_initializer(0.1))
        fc7_out = tf.matmul(hidden_fc7, weights)+biases
        y_out = tf.nn.softmax(fc7_out)
        scope.reuse_variables()

    return fc7_out, y_out

def inference_reuse(input_image, keep_prob):
    param_to_send = []
    with tf.variable_scope('conv1', reuse=True) as scope:
        weights = tf.get_variable("weights", shape=[11,11,3,96],initializer=tf.truncated_normal_initializer(stddev = 0.1))        
        biases = tf.get_variable('biases', shape=[96],initializer=tf.constant_initializer(0.1))
        x_image = tf.reshape(input_image, [-1,227,227,3])        
        hidden_c1 = tf.nn.relu(conv2d(x_image, weights, 'VALID')+biases)
        hidden_pool1 = max_pool_2x2(lrn(hidden_c1, 'n1'))
        scope.reuse_variables()
    with tf.variable_scope('conv2', reuse=True) as scope:
        weights = tf.get_variable("weights", shape=[5,5,48,256],initializer=tf.truncated_normal_initializer(stddev = 0.1))        
        biases = tf.get_variable('biases', shape=[256],initializer=tf.constant_initializer(0.1))
       
        group = 2
        input_groups = tf.split(3, group, hidden_pool1)
        kernel_groups = tf.split(3, group, weights)
        output_groups = [conv2d1s(i, k, 'SAME') for i,k in zip(input_groups, kernel_groups)]
        conv = tf.concat(3, output_groups)     

        hidden_c2 = tf.nn.relu(conv+biases)
        hidden_pool2 = max_pool_2x2(lrn(hidden_c2, 'n2'))
        scope.reuse_variables()

    with tf.variable_scope('conv3np', reuse=True) as scope:
        weights = tf.get_variable("weights", shape=[3,3,256,384],initializer=tf.truncated_normal_initializer(stddev = 0.1))        
        biases = tf.get_variable('biases', shape=[384],initializer=tf.constant_initializer(0.1))
       
        hidden_c3 = tf.nn.relu(conv2d1s(hidden_pool2, weights, 'SAME')+biases)
        scope.reuse_variables()

    with tf.variable_scope('conv4np', reuse=True) as scope:
        weights = tf.get_variable("weights", shape=[3,3,192,384],initializer=tf.truncated_normal_initializer(stddev = 0.1))        
        biases = tf.get_variable('biases', shape=[384],initializer=tf.constant_initializer(0.1))

        group = 2
        input_groups = tf.split(3, group, hidden_c3)
        kernel_groups = tf.split(3, group, weights)
        output_groups = [conv2d1s(i, k, 'SAME') for i,k in zip(input_groups, kernel_groups)]
        conv = tf.concat(3, output_groups)     
      
        hidden_c4 = tf.nn.relu(conv+biases)
        scope.reuse_variables()
    with tf.variable_scope('conv5', reuse=True) as scope:
        weights = tf.get_variable("weights", shape=[3,3,192,256],initializer=tf.truncated_normal_initializer(stddev = 0.1))        
        biases = tf.get_variable('biases', shape=[256],initializer=tf.constant_initializer(0.1))

        group = 2
        input_groups = tf.split(3, group, hidden_c4)
        kernel_groups = tf.split(3, group, weights)
        output_groups = [conv2d1s(i, k, 'SAME') for i,k in zip(input_groups, kernel_groups)]
        conv = tf.concat(3, output_groups)      
       
        hidden_c5 = tf.nn.relu(conv+biases)
        hidden_pool5 = max_pool_2x2(hidden_c5)
        scope.reuse_variables()

    with tf.variable_scope('fc6', reuse=True) as scope:
        weights = tf.get_variable("weights", shape=[6*6*256, 4096],initializer=tf.truncated_normal_initializer(stddev = 0.1))        
        biases = tf.get_variable('biases', shape=[4096],initializer=tf.constant_initializer(0.1))
        hidden_pool5_flat = tf.reshape(hidden_pool5, [-1, 6*6*256])
        hidden_fc6 = tf.nn.relu(tf.matmul(hidden_pool5_flat, weights)+biases)
    
        # Add dropout
        scope.reuse_variables()

    with tf.variable_scope('fc7', reuse=True) as scope:
        weights = tf.get_variable("weights", shape=[4096, 4096],initializer=tf.truncated_normal_initializer(stddev = 0.1))        
        biases = tf.get_variable('biases', shape=[4096],initializer=tf.constant_initializer(0.1))
       
        hidden_fc7_flat = tf.reshape(hidden_fc6, [-1, 4096])
        hidden_fc7 = tf.nn.relu(tf.matmul(hidden_fc7_flat, weights)+biases)
    
        # Add dropout
        scope.reuse_variables()

    with tf.variable_scope('softmax', reuse=True) as scope:
        weights = tf.get_variable("weights", shape=[4096, 1000],initializer=tf.truncated_normal_initializer(stddev = 0.1))        
        biases = tf.get_variable('biases', shape=[1000],initializer=tf.constant_initializer(0.1))
        fc7_out = tf.matmul(hidden_fc7, weights)+biases
        y_out = tf.nn.softmax(fc7_out)
        scope.reuse_variables()

    return fc7_out, y_out


