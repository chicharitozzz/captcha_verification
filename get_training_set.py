# @Time : 2018/10/25 17:00
# @Author : Chicharito_Ron
# @File : get_training_set.py
# @Software: PyCharm Community Edition

import requests
import random
import time
import os

if __name__ == '__main__':
    for i in range(1000):
        r = requests.get('http://zhixing.court.gov.cn/search/captcha.do')

        if r.content:
            with open('training_set/pic{}.png'.format(random.randint(1, 1e10)), 'wb') as f:
                f.write(r.content)

        time.sleep(3)

    print(len(os.listdir('training_set')))


