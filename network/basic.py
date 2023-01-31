import tensorflow as tf
from keras import layers

class BaseLayer(layers.Layer):
    def __init__(self, filters, size, padding, **kwargs) -> None:
        super(BaseLayer, self).__init__(**kwargs)
        self.Conv2D = layers.Conv2D(filters=filters, kernel_size=size, padding=padding)

    def call(self, x):
        x = self.Conv2D(x)
        x = tf.nn.relu(x)
        return x
