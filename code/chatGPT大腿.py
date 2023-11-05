import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def unet_model(input_shape):
    # 定义U-Net模型
    inputs = tf.keras.layers.Input(shape=input_shape)

    # 编码器
    conv1 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(conv3)
    pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)

    # 解码器
    up4 = tf.keras.layers.Conv2DTranspose(128, 2, strides=(2, 2), padding='same')(pool3)
    up4 = tf.keras.layers.concatenate([up4, conv3])
    conv4 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(up4)
    conv4 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(conv4)

    up5 = tf.keras.layers.Conv2DTranspose(64, 2, strides=(2, 2), padding='same')(conv4)
    up5 = tf.keras.layers.concatenate([up5, conv2])
    conv5 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(up5)
    conv5 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(conv5)

    up6 = tf.keras.layers.Conv2DTranspose(32, 2, strides=(2, 2), padding='same')(conv5)
    up6 = tf.keras.layers.concatenate([up6, conv1])
    conv6 = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(up6)
    conv6 = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(conv6)

    # 输出层
    outputs = tf.keras.layers.Conv2D(1, 1, activation='sigmoid')(conv6)

    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    return model


# 创建U-Net模型实例
input_shape = (256, 256, 1)
model = unet_model(input_shape)

# 打印模型概要
model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),optimizer="sgd",metrics=["accuracy"],)
model.summary()
tf.keras.utils.plot_model(model, to_file='model.png')
