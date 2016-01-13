import glob
import tensorflow as tf
import numpy as np
import re
import mnist_base as m_b
import input_data
import util_algebra as lalg
import util_logging as ul
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class PlotNNImages(object):
    def __init__(self, source_folder, target_folder):
        self.BATCH_SIZE = 256
        self.cur_batch=0

        # Read the data
        self.mnist_source = input_data.read_data_sets(source_folder, one_hot=True)
        self.mnist_target = input_data.read_data_sets(target_folder, one_hot=True)

    def plot_all_nn_images(self):
        existing_files = glob.glob('nn_list_a*.npy')
        for file_n in existing_files:
            after_name = file_n
            before_name = re.sub('list_a','list_b', after_name)
            batch_id = int(re.sub('.npy','',re.sub('nn_list_a','',after_name)))
            after_nns = np.load(after_name)
            before_nns = np.load(before_name)
            final_im = np.zeros((5,28*3))
            diff_im = np.zeros((5,28*3))
            for j in range(self.BATCH_SIZE):
                target_id = batch_id*self.BATCH_SIZE+j
                source_id_after = after_nns[j]
                source_id_before = before_nns[j]
                row_im = np.concatenate( (self.mnist_target.train.images[target_id].reshape((28, 28)), 
                                          self.mnist_source.train.images[source_id_before].reshape((28, 28)),
                                          self.mnist_source.train.images[source_id_after].reshape((28, 28)))
                                       , axis=1 )
                final_im = np.concatenate((final_im, row_im), axis=0)
                if not source_id_after == source_id_before:
                    diff_im = np.concatenate((diff_im, row_im), axis=0)
            mpimg.imsave('step_'+str(batch_id)+'_images.png', final_im, cmap=plt.cm.binary)
            mpimg.imsave('diff_step_'+str(batch_id)+'_images.png', diff_im, cmap=plt.cm.binary)


a = PlotNNImages('MNIST_a', 'MNIST_r')
a.plot_all_nn_images()
