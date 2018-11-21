# @Time : 2018/11/14 19:32 
# @Author : Chicharito_Ron
# @File : check_train_set.py 
# @Software: PyCharm Community Edition
import string
import os

chars = string.digits + string.ascii_lowercase
chars_len = len(chars)
captcha = 4
train_dir = 'F:/绝命之学习/验证码识别/training_set/'


if __name__ == '__main__':
    fname = os.listdir(train_dir)
    for f in fname:
        t = f.split('.')[0]
        if len(t) != 4:
            print(t)
        for i in t:
            if i not in chars:
                print(t)

