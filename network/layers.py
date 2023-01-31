from keras import layers
import tensorflow as tf
import sys
from tensorflow import nn
sys.path.append("../preprocessing/__init__.py")
from ..preprocessing import Base
sys.path.append("./basic.py")
from .basic import BaseLayer

class SmallBox(layers.Layer):
    def __init__(self, filters:int, size:int, padding:str, name, **kwargs) -> any:
        super(SmallBox, self).__init__(name=name, **kwargs)
        self.base_1 = BaseLayer(filters=filters, size=size, padding=padding)
        self.batchNormalization = layers.BatchNormalization()
        self.POOLER = layers.MaxPooling2D()

    def call(self, x):
        x = self.base_1(x)
        x = self.batchNormalization(x)
        x = self.POOLER(x)
        x = nn.relu(x)
        return x

class MediumBox(layers.Layer):
    def __init__(self, filters:int, size:int, padding:str, name, **kwargs):
        super(MediumBox, self).__init__(name=name, **kwargs)
        self.BASE = Base()
        self.base_1 = BaseLayer(filters=filters, size=size, padding=padding)
        self.base_2 = BaseLayer(filters=self.BASE.channels(filters), size=size, padding=padding)
        self.batchnormaliztion = layers.BatchNormalization()
        self.LAYER_NORMALIZATION = layers.LayerNormalization()
        self.POOLER = layers.MaxPooling2D()

    def call(self, x):
        x = self.base_1(x)
        x = self.POOLER(x)
        x = self.base_2(x)
        x = self.batchnormaliztion(x)
        x = self.POOLER(x)
        x = nn.relu(x)
        x = self.LAYER_NORMALIZATION(x)
        return x

class LargeBox(layers.Layer):
    def __init__(self, filters:int, size:int, padding:str, name, zeros=(1, 1), **kwargs):
        super(LargeBox, self).__init__(name=name, **kwargs)
        self.BASE = Base()
        self.base_1 = BaseLayer(filters=filters, size=size, padding=padding)
        self.base_2 = BaseLayer(filters=self.BASE.channels(filters), size=size, padding=padding)
        self.base_3 = BaseLayer(filters=512, size=self.BASE.filter_size(size), padding=padding)
        self.base_4 = BaseLayer(filters=filters*2, size=size, padding=padding)
        self.batchnormaliztion = layers.BatchNormalization()
        self.LAYER_NORMALIZATION = layers.LayerNormalization()
        self.zero = layers.ZeroPadding2D(padding=zeros)
        self.POOLER = layers.MaxPooling2D()

    def call(self, x):
        x = self.base_1(x)
        x = self.POOLER(x)
        x = self.base_2(x)
        x = self.batchnormaliztion(x)
        x = self.POOLER(x)
        x = self.base_3(x)
        x = self.POOLER(x)
        x = self.base_4(x)
        x = self.POOLER(x)
        x = self.zero(x)
        x = self.batchnormaliztion(x)
        x = nn.relu(x)
        x = self.LAYER_NORMALIZATION(x)

        return x

class BoxError(tf.errors.InternalError):...
