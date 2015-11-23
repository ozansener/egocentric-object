import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

dt = np.dtype(np.uint32).newbyteorder('>')

images = glob.glob('*.png')
labels = open('ground_truth.txt').read().split('\n')
labels.pop()

label_dict = {y[0]:int(y[1]) for y in map(lambda x:x.split(':'), labels)}

data_array = np.zeros([len(label_dict), 28, 28, 1], dtype=np.uint8)

for im_id, im in enumerate(label_dict):
    IM = mpimg.imread(im)*255.0
    data_array[im_id,:,:,0] = IM

array_buff = np.getbuffer(data_array)
fid = open('images.bn','wb')
fid.write(np.getbuffer(np.array([2051], dtype=dt)))
fid.write(np.getbuffer(np.array([len(label_dict)], dtype=dt)))
fid.write(np.getbuffer(np.array([28], dtype=dt)))
fid.write(np.getbuffer(np.array([28], dtype=dt)))
fid.write(array_buff)

label_buff = np.getbuffer(np.array(label_dict.values(), dtype=np.uint8))
fidl = open('labels.bn','wb')
fidl.write(np.getbuffer(np.array([2049], dtype=dt)))
fidl.write(np.getbuffer(np.array([len(label_dict)], dtype=dt)))
fidl.write(label_buff)
