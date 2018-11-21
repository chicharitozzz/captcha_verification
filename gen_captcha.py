# @Time : 2018/10/26 14:11 
# @Author : Chicharito_Ron
# @File : gen_captcha.py 
# @Software: PyCharm Community Edition
# 创建batch等


import string
import random
from PIL import Image
import numpy as np
import os

# 验证码中的字符
# chars = string.digits + string.ascii_letters
chars = string.digits + string.ascii_lowercase
chars_len = len(chars)
captcha = 4

#  验证码尺寸
img_height = 70
img_width = 160

#  训练集与测试集
train_dir = 'F:/绝命之学习/验证码识别/training_set/'
test_dir = 'F:/绝命之学习/验证码识别/testing_set/'
vert_dir = 'F:/绝命之学习/验证码识别/vert_set/'


def convert2grey(img):
    """转成灰度图像"""
    return img.convert('L')  # translating a color image to black and white


def rand_char():
    text = ''
    for _ in range(4):
        text += random.choice(chars)
    return text


def char2pos(c):
    """one-hot,返回字符的索引"""
    for i in range(chars_len):
        if chars[i] == c:
            return i
    return False


def text2vec(text):
    """验证码转向量, 向量包含顺序信息和字符信息。
    chars_len长度编码一个字符。"""
    vector = np.zeros(4 * chars_len)

    for i, c in enumerate(text):
        idx = i * chars_len + char2pos(c)
        vector[idx] = 1
    return vector


def vec2text(vec):
    """向量转验证码"""
    pos = vec.nonzero()[0].tolist()
    text = ''
    for i, idx in enumerate(pos):
        pos = idx - i * chars_len
        text += chars[pos]

    return text


def get_batch(batch_size=64):
    """生成训练batch"""
    capt_lists = os.listdir(train_dir)
    tmp_batch = np.random.choice(capt_lists, batch_size, replace=False)
    X = []
    Y = []

    for pic in tmp_batch:
        im = Image.open(train_dir + pic)
        text = pic.split('.')[0]

        x = np.array(im.convert('L')).flatten() / 255
        x = x.reshape((img_height, img_width, 1))

        y = text2vec(text)

        X.append(x)
        Y.append(y)

    return np.array(X), np.array(Y)


def get_testset(size=100):
    """测试集"""
    capt_lists = os.listdir(test_dir)
    tmp_batch = np.random.choice(capt_lists, size, replace=False)
    X = []
    Y = []

    for pic in tmp_batch:
        im = Image.open(test_dir + pic)
        text = pic.split('.')[0]

        x = np.array(im.convert('L')).flatten() / 255
        x = x.reshape((img_height, img_width, 1))

        y = text2vec(text)

        X.append(x)
        Y.append(y)

    return np.array(X), np.array(Y)


def gen_vertset(size=10):
    """验证集"""
    capt_lists = os.listdir(vert_dir)
    tmp_batch = np.random.choice(capt_lists, size, replace=False)
    X = []
    T = []

    for pic in tmp_batch:
        im = Image.open(vert_dir + pic)
        text = pic.split('.')[0]

        x = np.array(im.convert('L')).flatten() / 255
        x = x.reshape((img_height, img_width, 1))

        X.append(x)
        T.append(text)

    return X, T


if __name__ == '__main__':
    x, y = get_testset()
    print(x.shape)
    print(y.shape)
