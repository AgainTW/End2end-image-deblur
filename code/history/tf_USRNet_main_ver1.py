import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tf_USRNet_func import p2o

def re_img_shape(img_ori):
	img = np.array([img_ori[:,:,0].T, img_ori[:,:,1].T, img_ori[:,:,2].T])
	return img

def tf_USRNet_batch():
	loc = "C:/AG/course notes/111_2/deblur final project/dataset download/test_blur/test/test_blur/000/"
	first_flag = 0
	for i in range(10):
		# img = 高x寬x通道
		img_ori = cv2.imread(loc+str(i).zfill(8)+".png")
		img_norm = re_img_shape(img_ori)	

		if(first_flag==0):
			x = np.array([img_norm])
			first_flag = 1
		else:
			x = np.append(x,[img_norm], axis=0)
	return x


x = tf_USRNet_batch()

plt.imshow(x[0,0,:,:],"gray")
plt.show()

w, h = x.shape[-2:]
sf = 1
otf = p2o(x, (w*sf, h*sf))


otf_fs = tf.signal.fftshift(otf[0,0,:,:])
otf_log = tf.math.log(tf.math.abs(otf_fs))
otf = tf.signal.ifft2d(otf_fs)
otf = tf.math.abs(otf)
otf = tf.cast(otf, dtype=float)
plt.imshow(otf_log,"gray")
plt.show()