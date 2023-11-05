from keras.models import Model
from keras.layers import Input, Conv2D
from tensorflow import Tensor
from tensorflow.keras.layers import ReLU, BatchNormalization, Add

def relu_bn(inputs: Tensor) -> Tensor:
    relu = ReLU()(inputs)
    bn = BatchNormalization()(relu)
    return bn

def residual_block(x: Tensor, downsample: bool, filters: int, kernel_size: int = 3) -> Tensor:
    y = tf.keras.layers.Conv2D(kernel_size=kernel_size, strides= (1 if not downsample else 2), filters=filters, padding="same")(x)
    y = relu_bn(y)
    y = tf.keras.layers.Conv2D(kernel_size=kernel_size, strides=1, filters=filters, padding="same")(y)

    if downsample:
        x = tf.keras.layers.Conv2D(kernel_size=1, strides=2, filters=filters, padding="same")(x)
    out = Add()([x, y])
    out = relu_bn(out)
    return out


def get_ResNet(input_shape, output_shape):
    
    inputs = tf.keras.Input(shape=input_shape, name='input_layer')
    num_filters = 32

    x = BatchNormalization()(inputs)
    x = tf.keras.layers.Conv2D(num_filters, kernel_size=3, strides=1, padding="same")(x)
    x = relu_bn(x)
    
    num_blocks_list = [2, 5, 2]
    for i in range(len(num_blocks_list)):
      num_blocks = num_blocks_list[i]
      for j in range(num_blocks):
        x = residual_block(x, downsample=(j==0 and i!=0), filters=num_filters)
      num_filters *= 2
    
    x = tf.keras.layers.AveragePooling2D(4)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    outputs = tf.keras.layers.Dense(output_shape, activation='softmax')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="functional_api")

    return model
