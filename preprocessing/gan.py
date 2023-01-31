import tensorflow as tf
import numpy as np



class Gan(object):
    def __init__(self) -> None:
        pass
    @staticmethod
    def normalize(x, y_:float=127.5):
        x = tf.cast((x-y_) / y_, tf.float32)
        return x
