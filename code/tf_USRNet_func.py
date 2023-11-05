import numpy as np
import tensorflow as tf

def upsample(x, sf=3):
    '''應該是完成了QQ
    s-fold upsampler
    Upsampling the spatial size by filling the new entries with zeros
    x: tensor image, NxCxWxH
    '''
    st = 0
    z = np.zeros((x.shape[0], x.shape[1], x.shape[2]*sf, x.shape[3]*sf), dtype=np.float32)
##筆記## [..., 0::sf, 0::sf] : 「...」是指忽略不操作；「0::sf」是指從0開始每sf步取樣一次
    z[..., 0::sf, 0::sf] = x.numpy()
    tf.convert_to_tensor(z, dtype=tf.float32)
    return z

def downsample(x, sf=3):
    '''應該是完成了QQ
    s-fold downsampler
    Keeping the upper-left pixel for each distinct sfxsf patch and discarding the others
    x: tensor image, NxCxWxH
    '''
    st = 0
    return x[..., st::sf, st::sf]


def p2o(psf, shape):
    '''應該是完成了QQ
    Convert point-spread function to optical transfer function.
    otf = p2o(psf) computes the Fast Fourier Transform (FFT) of the
    point-spread function (PSF) array and creates the optical transfer
    function (OTF) array that is not influenced by the PSF off-centering.
    Args:
        psf: NxCxhxw
        shape: [H, W]
    Returns:
        otf: NxCxHxWx2
    '''
    otf = tf.zeros(psf.shape[:-2] + shape, dtype=tf.complex64)
    otf = tf.identity(psf)
    otf = tf.cast(otf, dtype=tf.complex64)
    otf = tf.roll(otf, shift=[-int(psf.shape[2]/2),-int(psf.shape[3]/2)], axis=[2,3])
    otf = tf.signal.fft2d(otf)
    return otf

def splits(a, sf):
    '''split a into sfxsf distinct blocks
    Args:
        a: NxCxWxH
        sf: split factor
    Returns:
        b: NxCx(W/sf)x(H/sf)x(sf^2)
    '''
    b = tf.split(a, sf, axis=2)
    b = tf.stack(b[:], axis=4)
    b = tf.split(b, sf, axis=3)
    b = tf.concat(b[:], axis=4)
    return b