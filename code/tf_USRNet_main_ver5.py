import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tf_USRNet_model
from tf_USRNet_model import USRNet
from tf_USRNet_func import p2o, upsample, downsample, splits

def re_img_shape(img_ori):
	img = np.array([img_ori[:,:,0], img_ori[:,:,1], img_ori[:,:,2]])
	return img

def tf_USRNet_batch(loc):
	first_flag = 0
	for i in range(1):
		# img = 高x寬x通道
		img_ori = cv2.imread(loc+str(i).zfill(8)+".png")
		img_gray = cv2.imread(loc+str(i).zfill(8)+".png", cv2.IMREAD_GRAYSCALE)
		img_gray = img_gray.reshape((img_gray.shape[0], img_gray.shape[1], 1))
		#img_norm = re_img_shape(img_ori)	

		if(first_flag==0):
			x = np.array([img_ori])
			k = np.array([img_gray])
			first_flag = 1
		else:
			x = np.append(x,[img_ori], axis=0)
			k = np.append(k,[img_gray], axis=0)
	return x, k

if __name__=='__main__':
	x_loc = "C:/AG/course notes/111_2/deblur final project/dataset download/test_blur/test/test_blur/000/"
	y_loc = "C:/AG/course notes/111_2/deblur final project/dataset download/val_sharp/val/val_sharp/000/"


	sf = np.array([1])
	x, k = tf_USRNet_batch(x_loc)
	y, _ = tf_USRNet_batch(y_loc)
	sigma = np.array([1])
	print(x.shape)
	print(y.shape)
	print(k.shape)

	'''x_size = x.shape[0]*x.shape[1]*x.shape[2]*x.shape[3]
	k_size = k.shape[0]*k.shape[1]*k.shape[2]*k.shape[3]
	x = x.reshape((x_size))
	k = k.reshape((k_size))
	x_size = np.array([x_size])
	k_size = np.array([k_size])
	x0 = np.concatenate([x_size, k_size, x ,k, sf, sigma]).reshape((1, -1))'''

	model = tf_USRNet_model.HyPaNet()
	model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=tf.keras.losses.MSE, metrics=['accuracy'])
	model.fit(k, y)
	model.summary()