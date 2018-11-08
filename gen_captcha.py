# @Time : 2018/10/26 14:11 
# @Author : Chicharito_Ron
# @File : gen_captcha.py 
# @Software: PyCharm Community Edition
# 生成验证码、创建batch等


import string
import random
from captcha.image import ImageCaptcha
from PIL import Image
import numpy as np
import os

# 验证码中的字符
chars = string.digits + string.ascii_letters
chars_len = len(chars)
captcha = 4

#  验证码尺寸
img_height = 70
img_width = 160

#  训练集与测试集
train_dir = 'F:/绝命之学习/验证码识别/my_captchas/'
test_dir = 'F:/绝命之学习/验证码识别/test_set/'


def convert2grey(img):
    """转成灰度图像"""
    return img.convert('L')  # translating a color image to black and white


def rand_char():
    text = ''
    for _ in range(4):
        text += random.choice(chars)
    return text


def gen_verification_code():
    """生成验证码"""
    img = ImageCaptcha(width=160, height=70)
    text = rand_char()
    captcha_image = img.generate_image(text).convert('L')
    captcha_image.save(test_dir + text + '.png')

    return True


def feature_vectors(img):
    """特征向量二值化"""
    x = []
    for i in range(img_height):
        for j in range(img_width):
            pixel = img.getpixel((j, i))
            if pixel == 255:
                x.append(1)
            else:
                x.append(0)

    return np.reshape(np.array(x), (img_height, img_width, 1))


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

        # x = feature_vectors(im)

        x = np.array(im).flatten() / 255
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

        # x = feature_vectors(im)

        x = np.array(im).flatten() / 255
        x = x.reshape((img_height, img_width, 1))

        y = text2vec(text)

        X.append(x)
        Y.append(y)

    return np.array(X), np.array(Y)


def gen_test_img():
    """测试"""
    img = ImageCaptcha(width=160, height=70)
    text = rand_char()

    captcha_image = img.generate_image(text)
    captcha_image.save('vert_set/' + text + '.png')
    captcha_image = captcha_image.convert('L')

    x = np.array(captcha_image.getdata()) / 255
    x = x.reshape((img_height, img_width, 1))

    return text, x


if __name__ == '__main__':
    for i in range(2000):
        gen_verification_code()

    # x, y = get_testset()
    # print(x.shape)
    # print(y.shape)

    # gen_test_img()
