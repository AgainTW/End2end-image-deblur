import cv2
import tensorflow as tf
import tensorflow.nn as nn
import tensorflow.keras.layers as layer
from keras.layers import Lambda

# -------------------------------------------------------
# return nn.Sequantial of (Conv + BN + ReLU)
# -------------------------------------------------------
def conv(in_channels=64, out_channels=64, kernel_size=(3,3), stride=(1,1), padding='same', bias=True, mode='CBR'):
	L = tf.keras.Sequential()
	for t in mode:
		if t == 'C':
			L.add(layer.Conv2D(filters=out_channels, kernel_size=kernel_size, strides=stride, padding=padding, use_bias=bias, data_format='channels_last'))
		elif t == 'T':
			L.add(layer.Conv2DTranspose(filters=out_channels, kernel_size=kernel_size, strides=stride, padding=padding, use_bias=bias, data_format='channels_last'))
		elif t == 'B':
			L.add(layer.BatchNormalization(momentum=0.9, epsilon=1e-04))
		elif t == 'R':
			L.add(layer.ReLU())
		elif t == 'L':
			L.add(layer.LeakyReLU(alpha=1e-1))
		elif t == '2':
			L.add(Lambda(lambda x:nn.depth_to_space(x, block_size=2, data_format='NCHW')))
		elif t == '3':
			L.add(Lambda(lambda x:nn.depth_to_space(x, block_size=3, data_format='NCHW')))
		elif t == '4':
			L.add(Lambda(lambda x:nn.depth_to_space(x, block_size=4, data_format='NCHW')))
		elif t == 'U':
			L.add(layer.UpSampling2D(size=(2,2), data_format='channels_first', interpolation='nearest'))
		elif t == 'u':
			L.add(layer.UpSampling2D(size=(3,3), data_format='channels_first', interpolation='nearest'))
		elif t == 'M':
			L.add(layer.MaxPool2D(pool_size=kernel_size, data_format='channels_last',  strides=stride))
		elif t == 'A':
			L.add(layer.AveragePooling2D(pool_size=kernel_size, data_format='channels_last', strides=stride))
		else:
			raise NotImplementedError('Undefined type: '.format(t))
	return L


'''
FFT block
'''
class FFTBlock(tf.keras.Model):
	def __init__(self, channel=64):
		super(FFTBlock, self).__init__()
		self.conv_fc = tf.keras.Sequential([
			layer.Conv2D(filters=1, kernel_size=(channel,channel), strides=(1,1), padding='same', use_bias=True, data_format='channels_first'),
			layer.ReLU(),
			layer.Conv2D(filters=channel, kernel_size=(1,1), strides=(1,1), padding='same', use_bias=True, data_format='channels_first'),
			Lambda(lambda x:tf.math.softplus(x))
			])

	def call(self, x, u, d, sigma):
		rho = self.conv_fc(sigma)
		x = u + tf.expand_dims(rho, -1)*tf.signal.rfft2d(x), d + self.real2complex(rho)
		x = self.divcomplex(x)
		x = tf.signal.irfft2d(x)
		return x

	def divcomplex(self, x, y):
		a = x[..., 0]
		b = x[..., 1]
		c = y[..., 0]
		d = y[..., 1]
		cd2 = c**2 + d**2
		return tf.stack([(a*c+b*d)/cd2, (b*c-a*d)/cd2], -1)

	def real2complex(self, x):
		return tf.stack([x, tf.zeros(x.shape).type_as(x)], -1)


# -------------------------------------------------------
# Concat the output of a submodule to its input
# -------------------------------------------------------
class ConcatBlock(tf.keras.Model):
	def __init__(self, submodule):
		super(ConcatBlock, self).__init__()
		self.sub = submodule

	def call(self, x):
		output = tf.cancat((x, self.sub(x)), axis=1)
		return output

	def __repr__(self):
		return self.sub.__repr__() + 'concat'


# -------------------------------------------------------
# Elementwise sum the output of a submodule to its input
# -------------------------------------------------------
class ShortcutBlock(tf.keras.Model):
	def __init__(self, submodule):
		super(ShortcutBlock, self).__init__()

		self.sub = submodule

	def call(self, x):
		output = x + self.sub(x)
		return output

	def __repr__(self):
		tmpstr = 'Identity + \n|'
		modstr = self.sub.__repr__().replace('\n', '\n|')
		tmpstr = tmpstr + modstr
		return tmpstr


# -------------------------------------------------------
# Res Block: x + conv(relu(conv(x)))
# -------------------------------------------------------
class ResBlock(tf.keras.Model):
	def __init__(self, in_channels=64, out_channels=64, kernel_size=(3,3), stride=(1,1), padding='same', bias=True, mode='CRC'):
		super(ResBlock, self).__init__()

		assert in_channels == out_channels, 'Only support in_channels==out_channels.'
		if mode[0] in ['R','L']:
			mode = mode[0].lower() + mode[1:]

		self.res = conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode)

	def call(self, x):
		res = self.res(x)
		return x + res


# -------------------------------------------------------
# Channel Attention (CA) Layer
# -------------------------------------------------------
def tf_AdaptiveAvgPool2d(input_size, output_size):
	stridesz = np.floor(input_size / output_size).astype(np.int32)
	kernelsz = input_size - (output_size - 1) * stridesz
	avg = layers.AveragePooling2D((kernelsz, kernelsz), strides=(stridesz, stridesz))
	return avg

# -------------------------------------------------------
# Channel Attention (CA) Layer
# -------------------------------------------------------
class CALayer(tf.keras.Model):
	def __init__(self, channel=64, reduction=16):
		super(CALayer, self).__init__()

		self.avg_pool = tf_AdaptiveAvgPool2d(1, 1)
		self.conv_fc = nn.Sequential(
			layer.Conv2D(filters=channel // reduction, kernel_size=(1,1), strides=stride, padding='same', use_bias=True, data_format='channels_first'),
			layer.ReLU(),
			layer.Conv2D(filters=channel, kernel_size=(1,1), strides=stride, padding='same', use_bias=True, data_format='channels_first'),
			tf.keras.activations.sigmoid()
			)

	def call(self, x):
		y = self.avg_pool(x)
		y = self.conv_fc(y)
		return x * y


# -------------------------------------------------------
# Residual Channel Attention Block (RCAB)
# -------------------------------------------------------
class RCABlock(tf.keras.Model):
	def __init__(self, in_channels=64, out_channels=64, kernel_size=(3,3), stride=(1,1), padding='same', bias=True, mode='CRC', reduction=16):
		super(RCABlock, self).__init__()
		assert in_channels == out_channels, 'Only support in_channels==out_channels.'
		if mode[0] in ['R','L']:
			mode = mode[0].lower() + mode[1:]

		self.res = conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode)
		self.ca = CALayer(out_channels, reduction)

	def call(self, x):
		res = self.res(x)
		res = self.ca(res)
		return res + x


# -------------------------------------------------------
# Residual Channel Attention Group (RG)
# -------------------------------------------------------
class RCAGroup(tf.keras.Model):
	def __init__(self, in_channels=64, out_channels=64, kernel_size=(3,3), stride=(1,1), padding='same', bias=True, mode='CRC', reduction=16, nb=12):
		super(RCAGroup, self).__init__()
		assert in_channels == out_channels, 'Only support in_channels==out_channels.'
		if mode[0] in ['R','L']:
			mode = mode[0].lower() + mode[1:]

		RG = tf.keras.Sequential()
		for _ in range(nb):
			RG.add(RCABlock(in_channels, out_channels, kernel_size, stride, padding, bias, mode, reduction))
		RG.add(conv(out_channels, out_channels, mode='C'))

		self.rg = tf.keras.Sequential(*RG)  # self.rg = ShortcutBlock(nn.Sequential(*RG))

	def call(self, x):
		res = self.rg(x)
		return res + x


# -------------------------------------------------------
# Residual Dense Block
# style: 5 convs
# -------------------------------------------------------
class ResidualDenseBlock_5C(tf.keras.Model):
	def __init__(self, nc=64, gc=32, kernel_size=(3,3), stride=(1,1), padding='same', bias=True, mode='CR'):
		super(ResidualDenseBlock_5C, self).__init__()

		# gc: growth channel
		self.conv1 = conv(nc, gc, kernel_size, stride, padding, bias, mode)
		self.conv2 = conv(nc+gc, gc, kernel_size, stride, padding, bias, mode)
		self.conv3 = conv(nc+2*gc, gc, kernel_size, stride, padding, bias, mode)
		self.conv4 = conv(nc+3*gc, gc, kernel_size, stride, padding, bias, mode)
		self.conv5 = conv(nc+4*gc, nc, kernel_size, stride, padding, bias, mode[:-1])

	def call(self, x):
		x1 = self.conv1(x)
		x2 = self.conv2(tf.cancat((x, x1), 1))
		x3 = self.conv3(tf.cancat((x, x1, x2), 1))
		x4 = self.conv4(tf.cancat((x, x1, x2, x3), 1))
		x5 = self.conv5(tf.cancat((x, x1, x2, x3, x4), 1))
		return tf.math.multiply(x5, tf.fill(x5.shape, 0.2)) + x


# -------------------------------------------------------
# Residual in Residual Dense Block
# 3x5c
# -------------------------------------------------------
class RRDB(tf.keras.Model):
	def __init__(self, nc=64, gc=32, kernel_size=(3,3), stride=(1,1), padding='same', bias=True, mode='CR'):
		super(RRDB, self).__init__()

		self.RDB1 = ResidualDenseBlock_5C(nc, gc, kernel_size, stride, padding, bias, mode)
		self.RDB2 = ResidualDenseBlock_5C(nc, gc, kernel_size, stride, padding, bias, mode)
		self.RDB3 = ResidualDenseBlock_5C(nc, gc, kernel_size, stride, padding, bias, mode)

	def call(self, x):
		out = self.RDB1(x)
		out = self.RDB2(out)
		out = self.RDB3(out)
		return tf.math.multiply(out, tf.fill(out.shape, 0.2)) + x


'''
# ======================
# Upsampler
# ======================
'''


# -------------------------------------------------------
# conv + subp + relu
# -------------------------------------------------------
def upsample_pixelshuffle(in_channels=64, out_channels=3, kernel_size=(3,3), stride=(1,1), padding='valid', bias=True, mode='2R'):
	assert len(mode)<4 and mode[0] in ['2', '3', '4'], 'mode examples: 2, 2R, 2BR, 3, ..., 4BR.'
	up1 = conv(in_channels, out_channels * (int(mode[0]) ** 2), kernel_size, stride, padding, bias, mode='C'+mode)
	return up1


# -------------------------------------------------------
# nearest_upsample + conv + relu
# -------------------------------------------------------
def upsample_upconv(in_channels=64, out_channels=3, kernel_size=(3,3), stride=(1,1), padding='valid', bias=True, mode='2R'):
	assert len(mode)<4 and mode[0] in ['2', '3'], 'mode examples: 2, 2R, 2BR, 3, ..., 3BR.'
	if mode[0] == '2':
		uc = 'UC'
	elif mode[0] == '3':
		uc = 'uC'
	mode = mode.replace(mode[0], uc)
	up1 = conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode=mode)
	return up1


# -------------------------------------------------------
# convTranspose + relu
# -------------------------------------------------------
def upsample_convtranspose(in_channels=64, out_channels=3, kernel_size=(2,2), stride=(2,2), padding='valid', bias=True, mode='2R'):
	assert len(mode)<4 and mode[0] in ['2', '3', '4'], 'mode examples: 2, 2R, 2BR, 3, ..., 4BR.'
	kernel_size = (int(mode[0]),int(mode[0]))
	stride = (int(mode[0]),int(mode[0]))
	mode = mode.replace(mode[0], 'T')
	up1 = conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode)
	return up1


'''
# ======================
# Downsampler
# ======================
'''


# -------------------------------------------------------
# strideconv + relu
# -------------------------------------------------------
def downsample_strideconv(in_channels=64, out_channels=64, kernel_size=(2,2), stride=(2,2), padding='valid', bias=True, mode='2R'):
	assert len(mode)<4 and mode[0] in ['2', '3', '4'], 'mode examples: 2, 2R, 2BR, 3, ..., 4BR.'
	kernel_size = int(mode[0])
	stride = int(mode[0])
	mode = mode.replace(mode[0], 'C')
	down1 = conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode)
	return down1


# -------------------------------------------------------
# maxpooling + conv + relu
# -------------------------------------------------------
def downsample_maxpool(in_channels=64, out_channels=64, kernel_size=(3,3), stride=(1,1), padding='valid', bias=True, mode='2R'):
	assert len(mode)<4 and mode[0] in ['2', '3'], 'mode examples: 2, 2R, 2BR, 3, ..., 3BR.'
	kernel_size_pool = int(mode[0])
	stride_pool = int(mode[0])
	mode = mode.replace(mode[0], 'MC')
	L = tf.keras.Sequential()
	L.add(conv(kernel_size=kernel_size_pool, stride=stride_pool, mode=mode[0]))
	L.add(conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode=mode[1:]))
	return L


# -------------------------------------------------------
# averagepooling + conv + relu
# -------------------------------------------------------
def downsample_avgpool(in_channels=64, out_channels=64, kernel_size=(3,3), stride=(1,1), padding='valid', bias=True, mode='2R'):
	assert len(mode)<4 and mode[0] in ['2', '3'], 'mode examples: 2, 2R, 2BR, 3, ..., 3BR.'
	kernel_size_pool = int(mode[0])
	stride_pool = int(mode[0])
	mode = mode.replace(mode[0], 'AC')
	L = tf.keras.Sequential()
	L.add(conv(kernel_size=kernel_size_pool, stride=stride_pool, mode=mode[0]))
	L.add(conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode=mode[1:]))
	return L


'''
# ======================
# NonLocalBlock2D: 
# embedded_gaussian
# +W(softmax(thetaXphi)Xg)
# ======================
'''


# -------------------------------------------------------
# embedded_gaussian
# -------------------------------------------------------
class NonLocalBlock2D(tf.keras.Model):
	def __init__(self, nc=64, kernel_size=(1,1), stride=(1,1), padding='same', bias=True, act_mode='B'
		, downsample=False, downsample_mode='maxpool'):

		super(NonLocalBlock2D, self).__init__()

		inter_nc = nc // 2
		self.inter_nc = inter_nc
		self.W = conv(inter_nc, nc, kernel_size, stride, padding, bias, mode='C'+act_mode)
		self.theta = conv(nc, inter_nc, kernel_size, stride, padding, bias, mode='C')

		if downsample:
			if downsample_mode == 'avgpool':
				downsample_block = downsample_avgpool
			elif downsample_mode == 'maxpool':
				downsample_block = downsample_maxpool
			elif downsample_mode == 'strideconv':
				downsample_block = downsample_strideconv
			else:
				raise NotImplementedError('downsample mode [{:s}] is not found'.format(downsample_mode))
			self.phi = downsample_block(nc, inter_nc, kernel_size, stride, padding, bias, mode='2')
			self.g = downsample_block(nc, inter_nc, kernel_size, stride, padding, bias, mode='2')
		else:
			self.phi = conv(nc, inter_nc, kernel_size, stride, padding, bias, mode='C')
			self.g = conv(nc, inter_nc, kernel_size, stride, padding, bias, mode='C')

	def call(self, x):
		'''
		:param x: (b, c, t, h, w)
		:return:
		'''

		batch_size = x.shape[0].value

		g_x = tf.reshape(self.g(x), (batch_size, self.inter_nc, -1))
		g_x = tf.transpose(g_x ,perm=[0, 2, 1])

		theta_x = tf.reshape(self.theta(x), (batch_size, self.inter_nc, -1))
		theta_x = tf.transpose(theta_x ,perm=[0, 2, 1])
		phi_x = tf.reshape(self.phi(x), (batch_size, self.inter_nc, -1))
		f = tf.matmul(theta_x, phi_x)
		f_div_C = nn.softmax(f, axis=-1)

		y = tf.matmul(f_div_C, g_x)
		y = tf.transpose(y ,perm=[0, 2, 1])
		y = tf.reshape(y, (batch_size, self.inter_nc, *x.size()[2:]))
		W_y = self.W(y)
		z = W_y + x

		return z


if __name__=='__main__':
	print(0)
