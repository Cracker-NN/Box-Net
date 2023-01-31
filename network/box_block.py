import tensorflow as tf
from keras import layers
import sys
import numpy as np
sys.path.append("../preprocessing/__init__.py")
from ..preprocessing import Base
sys.path.append("./basic.py")
from .basic import BaseLayer
sys.path.append("./layers.py")
from .layers import BoxError, SmallBox, MediumBox, LargeBox



class BoxBlock(layers.Layer):
    def __init__(self, filters: list, size: list, padding: str, **kwargs):
        super(BoxBlock, self).__init__(**kwargs)
        self.BASE = Base()
        self.BOX_1 = SmallBox(filters=filters[0], size=size[0], padding=padding, name='BOX_1')
        self.BOX_2 = MediumBox(filters=filters[1], size=size[1], padding=padding, name='BOX_2')
        self.BOX_3 = SmallBox(filters=filters[2], size=size[2], padding=padding, name='BOX_3')
        self.BOX_4 = BaseLayer(filters=int(np.max(filters)), size=int(np.min(size)), padding=padding, name='BOX_4')
        self.BATCH_NORMALIZTION = layers.BatchNormalization()
        self.LAYER_NORMALIZTION = layers.LayerNormalization()
        self.ZERO_PADDING = layers.ZeroPadding2D()
        self.LEARNING_RATE = self.BASE.LEARNING_RATE

    def call(self, x):
        x = self.BOX_1(x)
        x = self.BOX_2(x)
        x = self.BATCH_NORMALIZTION(x)
        x = self.BOX_3(x)
        x = self.BOX_4(x)
        x = self.ZERO_PADDING(x)
        x = tf.nn.relu(x)
        x = self.LAYER_NORMALIZTION(x)

        return x
