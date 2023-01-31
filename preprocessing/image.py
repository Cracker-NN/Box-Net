# _*_ utf-8 _*_
# @uthor : Aman Raj
# Filename : augmentation.py
# File Modified : 30/01/2023

import tensorflow as tf
import numpy as np
import cv2
import os
import shutil
import time
import requests
from bs4 import BeautifulSoup
from urllib.request import urlretrieve
import math
import imghdr


class ImagePre(object):
    def __init__(self) -> None:
        super().__init__()

    def aug(self, img, save_dir, name, total: int = 20) -> None:
        gen = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=int(np.random.randint(10, 20, size=1)),
            shear_range=float(
                "0." + str(int(np.random.randint(1, 5, size=1)))),
            zoom_range=float("0." + str(int(np.random.randint(1, 9, size=1)))),
            horizontal_flip=True,
            fill_mode='nearest'
        )

        img = cv2.imread(img)
        img = cv2.resize(img, (256, 256))
        img_size = img.reshape(1, 256, 256, 3)
        if os.path.isdir(save_dir):
            save_dir = save_dir
        else:
            os.mkdir(save_dir)
            save_dir = save_dir
        count = 0
        for _ in gen.flow(img_size, batch_size=1, save_to_dir=save_dir, save_prefix=name, save_format='jpeg'):
            count = count + 1
            if count == total:
                break
        return "\033[1;32mAugmentation Is Completed\033[0m"


    def image_split(self, to, from_, size=0.2, shuffle=True) -> None:
        file = []
        for i in os.listdir(to):
            file.append(os.path.join(to, i))

        if shuffle:
            np.random.shuffle(file)

        sizes = int(np.round(float(np.multiply(size, int(len(file))))))
        for i in range(sizes):
            shutil.move(file[int(i)], from_)

        return "Folder is Splited"


    def download(self, url,  no: int = 5, save_format: str = 'jpeg', save_prefix: str = 'image', save_dir: str = 'image') -> None:
        URL = str(url)
        USER_AGENT = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.83 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3',
            'Accept-Encoding': 'none',
            'Accept-Language': 'en-US,en;q=0.8',
            'Connection': 'keep-alive',
        }
        IMG_NUMBER = int(no)

        if os.path.isdir(save_dir):
            SAVE_DIR = save_dir
        else:
            os.mkdir(save_dir)
            SAVE_DIR = save_dir

        FORMAT = save_format
        PREFIX = save_prefix

        response = requests.get(URL, headers=USER_AGENT)
        soup = BeautifulSoup(response.text, 'html.parser')
        count = 0
        links = []
        for i in soup.findAll('img', {'class': 'rg_i Q4LuWd'}):
            try:
                key = i['data-src']
                links.append(key)
                count += 1
                if count >= IMG_NUMBER:
                    break
            except KeyError:
                continue

        starting_time = int(time.strftime("%S"))
        print("\033[92mDownloading is Started...")
        img_count = 0
        for img in links:
            img_count += 1
            urlretrieve(img, os.path.join(
                f"{SAVE_DIR}", f"{PREFIX}({img_count}).{FORMAT}"))

        print("\033[91mImages is Downloaded")
        final_time = int(int(time.strftime('%S'))-starting_time)
        print(
            f"\033[1;33mTotal Time Taken : \033[1;36m{math.gcd(final_time)}s\033[0m")


    def image_to_array(self, path, dtype:str='float64', resize: tuple = (256, 256), channel: int = 3):
        img = cv2.imread(path)
        img = cv2.resize(img, resize)
        img = np.reshape(img, tuple([1] + list(resize) + list(channel)))
        img = img.astype(dtype)
        img = img / 255.
        return img


    def img_cleaner(self, img_path) -> str:
        ext = ['jpg', 'jpeg', 'png', 'bmp', 'gif']
        try:
            what = imghdr.what(img_path)
            if what not in ext:
                os.remove(img_path)
                return f'Image Path {img_path} Removed'
        except Exception:
            return f'An Issued Found !!'

    def normalize(self, x, labels):
        x = tf.cast(x/255., tf.float32)
        return x, labels
