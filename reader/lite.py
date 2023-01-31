import tensorflow as tf
import sys
sys.path.append("../preprocessing/__init__.py")
from ..preprocessing import ImagePre

class CNN_InterPreter(object):
    def __init__(self, model_path:str, dtype='float32', image_width=256, image_height=256, channels=3) -> None:
        super().__init__()
        self.D_TYPE = dtype
        self.Image = ImagePre()
        self.WIDTH = image_width
        self.HEIGHT = image_height
        self.CHANNEL = channels
        self.INTERPRETER = tf.lite.Interpreter(model_path)
        self.INTERPRETER.allocate_tensors()
        self.INPUT = self.INTERPRETER.get_input_details()
        self.OUTPUT = self.INTERPRETER.get_output_details()

    def predict(self, image_path:str):
        image = self.Image.image_to_array(image_path, dtype=self.D_TYPE, resize=(self.HEIGHT, self.WIDTH), channel=self.CHANNEL)
        self.INTERPRETER.set_tensor(self.INPUT[0]['index'], image)
        self.INTERPRETER.invoke()
        pred = self.INTERPRETER.get_tensor(self.OUTPUT[0]['index'])

        return pred

class Interpreter(object):
    def __init__(self, model_path:str, dtype='float32') -> None:
        self.INTERPRETER = tf.lite.Interpreter(model_path)
        self.INTERPRETER.allocate_tensors()
        self.INPUT = self.INTERPRETER.get_input_details()
        self.OUTPUT = self.INTERPRETER.get_output_details()

    def predict(self, input):
        self.INTERPRETER.set_tensor(self.INPUT[0]['index'], input)
        self.INTERPRETER.invoke()
        pred = self.INTERPRETER.get_tensor(self.OUTPUT[0]['index'])
        return pred
