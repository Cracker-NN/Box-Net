import sys
import tensorflow as tf
from keras import layers

sys.path.append("./box_block.py")
from .box_block import BoxBlock


class BoxNet(layers.Layer):
    def __init__(self, **kwargs):
        super(BoxNet).__init__(**kwargs)
        self.LEARNING_RATE = 3e-4
        self.Block = BoxBlock(filters=[64, 64, 32], size=[7, 3, 3], padding='same')
        self.EXT_CONV2D = layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same')
        self.BATCH_NORMALIZATION = layers.BatchNormalization()
        self.LAYER_NORMALIZATION = layers.LayerNormalization()
        self.POOLER = layers.MaxPooling2D()
        self.AVERAGE_POOLER_GLOBAL = layers.GlobalAveragePooling2D()

    def call(self, x):
        x = self.Block(x)
        x = tf.nn.leaky_relu(x)
        x = self.BATCH_NORMALIZATION(x)
        x = self.EXT_CONV2D(x)
        x = self.AVERAGE_POOLER_GLOBAL(x)
        x = tf.nn.relu(x)
        x = self.LAYER_NORMALIZATION(x)

        return x
