# @Time : 2018/11/6 10:42 
# @Author : Chicharito_Ron
# @File : interface.py 
# @Software: PyCharm Community Edition

from flask import Flask, request
from flask_cors import CORS
from flask_restful import Resource, Api
from PIL import Image
import numpy as np
import string
import captcha_verify

# 验证码中的字符
chars = string.digits + string.ascii_letters
chars_len = len(chars)
captcha = 4

# 图片尺寸
img_height = 70
img_width = 160

app = Flask(__name__)
CORS(app, supports_credentials=True)  # 跨域
api = Api(app)


def ident(fig):
    im = Image.open(fig)
    pixes = np.array(im)

    # 判断图片尺寸, 是否需要转成灰图
    if pixes.shape == (img_height, img_width):
        pass
    elif pixes.shape[0] != img_height or pixes.shape[1] != img_width:
        return '图片尺寸错误'
    else:
        pixes = np.array(im.convert('L'))

    x = (pixes / 255).reshape((img_height, img_width, 1))

    cv = captcha_verify.Captcha_Verify()
    p_text = cv.identification(x)

    return p_text


class Captcha(Resource):
    def get(self):
        return 'captcha vertification interface'

    def post(self):
        fig = request.files['fig']  # 获取文件对象
        p_text = ident(fig)

        return p_text


api.add_resource(Captcha, '/captcha')

if __name__ == '__main__':
    app.run(port=1234, debug=True)
