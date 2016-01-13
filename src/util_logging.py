import json
import numpy as np

class DeepLogger:
    def __init__(self):
        self.file_name = 'test_acc_feature_log.t'
    def add_log(self, line):
        f = open(self.file_name, 'a')
        f.write(line+'\n')
        f.close()

    def write_nn(self, nns, idd, is_second):
        if is_second:
            np.save('nn_list_a'+str(idd), nns)
        else:
            np.save('nn_list_b'+str(idd), nns)
 
