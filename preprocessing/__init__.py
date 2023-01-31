# _*_ utf-8 _*_
# @uthor : Aman Raj
# Filename : augmentation.py
# File Modified : 30/01/2023

import sys
import os
import numpy as np
from PIL import Image
import zipfile
sys.path.append("./image.py")
from .image import ImagePre
sys.path.append("./gan.py")
from .gan import Gan

class Base(object):
    def __init__(self) -> None:
        self.LEARNING_RATE = 3e-4
        self.USER_AGENT = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.83 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3',
            'Accept-Encoding': 'none',
            'Accept-Language': 'en-US,en;q=0.8',
            'Connection': 'keep-alive',
        }

    def mkdir(self, path) -> str:
        os.makedirs(name=path, exist_ok=True)
        return f'"{path}" Created !!'

    def channels(self, x, scalar_val: int = 7) -> any:
        x = np.divide(np.power(x, 2), scalar_val)
        x = int(np.round(x))
        return x

    def filter_size(self, x: int):
        x = int(np.round(np.divide(x, 2)))
        x = x + 1
        if x > 3:
            return int(np.subtract(x, 1))
        else:
            return int(np.add(x, 1))

    def array_to_image(self, image) -> np.ndarray:
        return Image.fromarray(image)

    def units(self, x: int, sum_val: int = 22, power: int = 2) -> int:
        x = np.add(x, + sum_val)
        return int(np.power(x, power))

    def _typo(self, x):
        return max(0, x)

    def _directory_maker(self, path: str, name: str) -> str:
        for i in os.listdir(path):
            self.mkdir(f"{name}/{i}")

    def extract(self, path) -> None:
        zips = zipfile.ZipFile(str(path), 'r')
        zips.extractall()
        zips.close()

    def _sigmoid(self, x):
        if float(x) < 50:
            return False    # class = 0
        else:
            return True     # class = 1

    def _softmax(self, x):
        x = np.argmax(x)
        return np.asarray(x)

    def _even(self, x):
        x = np.divide(x, 2)
        x = str(int(x))
        if x.endswith(".0"):
            return True
        else:
            return False

    def units(self, x):
        return int(int(np.power(x, 2)) // 2)
