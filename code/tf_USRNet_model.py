import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layer
import tf_USRNet_basicblock as B
from tf_USRNet_func import p2o, upsample
from keras.layers import Lambda

# tensorflow的專家模式
class ResUNet(tf.keras.Model):
	def __init__(self, in_nc=4, out_nc=3, nc=[32, 64, 128, 256], nb=2, act_mode='R', downsample_mode='strideconv', upsample_mode='convtranspose'):
		super(ResUNet, self).__init__()

		self.m_head = B.conv(in_nc, nc[0], bias=False, mode='C', padding='same')

		# downsample
		if downsample_mode == 'avgpool':
			downsample_block = B.downsample_avgpool
		elif downsample_mode == 'maxpool':
			downsample_block = B.downsample_maxpool
		elif downsample_mode == 'strideconv':
			downsample_block = B.downsample_strideconv
		else:
			# raise：實現報錯功能		;	{:s}：format格式化輸出	；	NotImplementedError
			raise NotImplementedError('downsample mode [{:s}] is not found'.format(downsample_mode))

		self.m_down1 = tf.keras.Sequential()
		self.m_down2 = tf.keras.Sequential() 
		self.m_down3 = tf.keras.Sequential()
		self.m_body = tf.keras.Sequential() 
		for _ in range(nb):
			self.m_down1.add(B.ResBlock(nc[0], nc[0], bias=False, mode='C'+act_mode+'C'))
			self.m_down2.add(B.ResBlock(nc[1], nc[1], bias=False, mode='C'+act_mode+'C'))
			self.m_down3.add(B.ResBlock(nc[2], nc[2], bias=False, mode='C'+act_mode+'C'))
			self.m_body.add(B.ResBlock(nc[3], nc[3], bias=False, mode='C'+act_mode+'C'))

		self.m_down1.add(downsample_block(nc[0], nc[1], bias=False, mode='2'))
		self.m_down2.add(downsample_block(nc[1], nc[2], bias=False, mode='2'))
		self.m_down3.add(downsample_block(nc[2], nc[3], bias=False, mode='2'))

		# upsample
		if upsample_mode == 'upconv':
			upsample_block = B.upsample_upconv
		elif upsample_mode == 'pixelshuffle':
			upsample_block = B.upsample_pixelshuffle
		elif upsample_mode == 'convtranspose':
			upsample_block = B.upsample_convtranspose
		else:
			raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))

		self.m_up3 = tf.keras.Sequential(upsample_block(nc[3], nc[2], bias=False, mode='2'))
		self.m_up2 = tf.keras.Sequential(upsample_block(nc[2], nc[1], bias=False, mode='2')) 
		self.m_up1 = tf.keras.Sequential(upsample_block(nc[1], nc[0], bias=False, mode='2'))
		for _ in range(nb):
			self.m_up3.add(B.ResBlock(nc[2], nc[2], bias=False, mode='C'+act_mode+'C'))
			self.m_up2.add(B.ResBlock(nc[1], nc[1], bias=False, mode='C'+act_mode+'C'))
			self.m_up1.add(B.ResBlock(nc[0], nc[0], bias=False, mode='C'+act_mode+'C'))
		self.m_tail = B.conv(nc[0], out_nc, bias=False, mode='C', padding='same')		

	def call(self, x):
		h, w = x.shape[-2:]
		paddingBottom = int(np.ceil(h/8)*8-h)
		paddingRight = int(np.ceil(w/8)*8-w)
		x = tf.pad(x, [[0,0],[0,0],[1,0],[1,0]], "SYMMETRIC")

		x = tf.cast(x, dtype=tf.float32)
		x1 = self.m_head(x)
		x2 = self.m_down1(x1)
		x3 = self.m_down2(x2)
		x4 = self.m_down3(x3)
		x = self.m_body(x4)
		x = self.m_up3(x+x4)
		x = self.m_up2(x+x3)
		x = self.m_up1(x+x2)
		x = self.m_tail(x)

		x = x[..., :h, :w]

		return x


"""
# --------------------------------------------
# (2) Data module, closed-form solution
# It is a trainable-parameter-free module  ^_^
# z_k = D(x_{k-1}, s, k, y, alpha_k)
# some can be pre-calculated
# --------------------------------------------
"""


class DataNet(tf.keras.Model):
	def __init__(self):
		super(DataNet, self).__init__()

	def call(self, x, FB, FBC, F2B, FBFy, alpha, sf):

		FR = FBFy + tf.signal.fft2d(alpha*x)
		x1 = tf.multiply(FB, FR)
		FBR = tf.reduce_mean(tf.split(x1, sf, axis=-1), axis=-1)
		invW = tf.reduce_mean(tf.split(F2B, sf, axis=-1), axis=-1)
		invWBR = tf.divide(FBR, invW + alpha)

		repeated_invWBR = tf.repeat(invWBR, sf, axis=2)
		repeated_invWBR = tf.repeat(repeated_invWBR, sf, axis=3)
		FCBinvWBR = FBC * repeated_invWBR

		FX = (FR-FCBinvWBR)/alpha
		Xest = tf.math.real(tf.signal.ifft2d(FX))

		return Xest


"""
# --------------------------------------------
# (3) Hyper-parameter module
# --------------------------------------------
"""


class HyPaNet(tf.keras.Model):
	def __init__(self, in_nc=2, out_nc=3, channel=64):
		super(HyPaNet, self).__init__()
		self.mlp = tf.keras.Sequential([
			layer.Conv2D(channel, (1,1), padding='valid', use_bias=True),
			layer.ReLU(),
			layer.Conv2D(channel, (1,1), padding='valid', use_bias=True),
			layer.ReLU(),
			layer.Conv2D(out_nc, (1,1), padding='valid', use_bias=True),
			Lambda(lambda x:tf.math.softplus(x))])

	def call(self, x):
		x = tf.cast(x, dtype='float32')
		x = self.mlp(x) + 1e-6
		return x


"""
# --------------------------------------------
# main USRNet
# deep unfolding super-resolution network
# --------------------------------------------
"""


class USRNet(tf.keras.Model):
	def __init__(self, n_iter=8, h_nc=64, in_nc=4, out_nc=3, nc=[64, 128, 256, 512], nb=2, act_mode='R', downsample_mode='strideconv', upsample_mode='convtranspose'):
		super(USRNet, self).__init__()

		self.d = DataNet()
		self.p = ResUNet(in_nc=in_nc, out_nc=out_nc, nc=nc, nb=nb, act_mode=act_mode, downsample_mode=downsample_mode, upsample_mode=upsample_mode)
		self.h = HyPaNet(in_nc=2, out_nc=n_iter*2, channel=h_nc)
		self.n = n_iter

	def call(self, x0):
		'''
		x: tensor, NxCxWxH
		k: tensor, Nx(1,3)xwxh
		sf: integer, 1
		sigma: tensor, Nx1x1x1
		'''
		x0 = x0[0]
		x_size = x0.eval()
		print("H"*30)
		print(x_size)
		k_size = x0[1]
		x = tf.reshape(x0[2:x_size+2], (1, 720, 1280, 3))
		k = tf.reshape(x0[x_size+2+1:x_size+2+1+k_size], (1, 720, 1280, 1))
		sf = x0[-2]
		sigma = x0[-1:].reshape((1,1,1,1))

		# initialization & pre-calculation
		w, h = x.shape[-2:]
		FB = p2o(k, (w*sf, h*sf))
		FBC = tf.math.conj(FB)
		F2B = tf.math.pow(tf.math.abs(FB), 2)
		STy = upsample(x, sf=sf)
		FBFy = FBC*tf.signal.fft2d(STy)
		x = tf.image.resize(x, size=(int(x.shape[2]*sf), int(x.shape[3]*sf)), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

		# hyper-parameter, alpha & beta
		sf_tensor = tf.cast(sf, dtype=sigma.dtype)
		sf_tensor = tf.expand_dims(sf_tensor, axis=0)
		sf_tensor = tf.tile(sf_tensor, [sigma.shape[0], 1])
		ab = self.h(tf.concat([sigma, sf_tensor], axis=1))

		# unfolding
		for i in range(self.n):
			x = self.d(x, FB, FBC, F2B, FBFy, ab[:, i:i+1, ...], sf)
			ab_repeated = tf.repeat(ab[:, i+self.n:i+self.n+1, ...], repeats=x.shape[1], axis=1)
			ab_repeated = tf.repeat(ab_repeated, repeats=x.shape[2], axis=2)
			ab_repeated = tf.repeat(ab_repeated, repeats=x.shape[3], axis=3)			
			x = self.p(tf.concat([x, ab_repeated], axis=1))

		return x


if __name__=='__main__':
	print(0)