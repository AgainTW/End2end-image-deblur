import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tf_USRNet_func import p2o, upsample

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
w, h = x.shape[-2:]
sf = 3

#z = upsample(x, sf=3)
img1 = x[0,0,:,:]
img2 = img1[0::10, 0::10]
plt.imshow(img1,"gray")
plt.show()
plt.imshow(img2,"gray")
plt.show()


