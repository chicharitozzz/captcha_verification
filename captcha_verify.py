# @Time : 2018/10/25 11:13
# @Author : Chicharito_Ron
# @File : captcha_verify.py 
# @Software: PyCharm Community Edition

import numpy as np
import tensorflow as tf
import string
from PIL import Image
import gen_captcha

# 验证码中的字符
chars = string.digits + string.ascii_letters
chars_len = len(chars)
captcha = 4

# 图片尺寸
img_height = 70
img_width = 160


class Captcha_Verify:
    def __init__(self):
        self.X = tf.placeholder(tf.float32, [None, img_height, img_width, 1])
        self.Y = tf.placeholder(tf.float32, [None, captcha * chars_len])
        self.prob = tf.placeholder(tf.float32)  # dropout

    def captcha_cnn(self, w_alpha=0.01, b_alpha=0.1):
        """网络模型"""
        w_c1 = tf.Variable(w_alpha * tf.random_normal([3, 3, 1, 32]))
        b_c1 = tf.Variable(b_alpha * tf.random_normal([32]))
        h_conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(self.X, w_c1, strides=[1, 1, 1, 1], padding='SAME'), b_c1))
        h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        h_dropout1 = tf.nn.dropout(h_pool1, self.prob)

        conv_width = np.ceil(img_width / 2)
        conv_height = np.ceil(img_height / 2)

        w_c2 = tf.Variable(w_alpha * tf.random_normal([3, 3, 32, 64]))
        b_c2 = tf.Variable(b_alpha * tf.random_normal([64]))
        h_conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(h_dropout1, w_c2, strides=[1, 1, 1, 1], padding='SAME'), b_c2))
        h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        h_dropout2 = tf.nn.dropout(h_pool2, self.prob)

        conv_width = np.ceil(conv_width / 2)
        conv_height = np.ceil(conv_height / 2)

        w_c3 = tf.Variable(w_alpha * tf.random_normal([3, 3, 64, 64]))
        b_c3 = tf.Variable(b_alpha * tf.random_normal([64]))
        h_conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(h_dropout2, w_c3, strides=[1, 1, 1, 1], padding='SAME'), b_c3))
        h_pool3 = tf.nn.max_pool(h_conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        h_dropout3 = tf.nn.dropout(h_pool3, self.prob)
        conv_width = np.ceil(conv_width / 2)
        conv_height = np.ceil(conv_height / 2)

        # Fully connected layer
        conv_width = int(conv_width)
        conv_height = int(conv_height)
        w_d = tf.Variable(w_alpha * tf.random_normal([conv_height * conv_width * 64, 1024]))
        b_d = tf.Variable(b_alpha * tf.random_normal([1024]))
        dense = tf.reshape(h_dropout3, [-1, 64*conv_width*conv_height])
        dense = tf.nn.relu(tf.add(tf.matmul(dense, w_d), b_d))
        dense = tf.nn.dropout(dense, self.prob)

        w_out = tf.Variable(w_alpha * tf.random_normal([1024, captcha * chars_len]))
        b_out = tf.Variable(b_alpha * tf.random_normal([captcha * chars_len]))
        out = tf.add(tf.matmul(dense, w_out), b_out)

        return out

    def training_cnn(self):
        """网络训练"""
        output = self.captcha_cnn()

        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output, labels=self.Y))

        optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

        predict = tf.reshape(output, [-1, captcha, chars_len])
        max_idx_p = tf.argmax(predict, 2)
        max_idx_l = tf.argmax(tf.reshape(self.Y, [-1, captcha, chars_len]), 2)
        correct_pred = tf.equal(max_idx_p, max_idx_l)
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        saver = tf.train.Saver(max_to_keep=3)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            # saver.restore(sess, './model/captcha_model')  # 继续训练
            step = 0
            while True:
                x_train, y_train = gen_captcha.get_batch(128)
                _, loss_ = sess.run([optimizer, loss], feed_dict={self.X: x_train, self.Y: y_train, self.prob: 0.75})
                print('第{}次迭代，损失函数值{}'.format(step, loss_))

                # 每100 step计算一次准确率
                if step % 100 == 0:
                    x_test, y_test = gen_captcha.get_testset()
                    acc = sess.run(accuracy, feed_dict={self.X: x_test, self.Y: y_test, self.prob: 1})
                    print('第{}次迭代，准确率{}'.format(step, acc))
                    saver.save(sess, "./model/captcha_model", global_step=None, write_meta_graph=True)

                    if acc > 0.9:
                        # saver.save(sess, "./model/captcha_model", global_step=None, write_meta_graph=True)
                        break

                step += 1

    def captcha_identification(self):
        """识别验证码测试"""
        output = self.captcha_cnn()
        saver = tf.train.Saver()
        # saver = tf.train.import_meta_graph('./model/captcha_model.meta')

        with tf.Session() as sess:
            saver.restore(sess, tf.train.latest_checkpoint('./model'))

            # sess.run(tf.global_variables_initializer())
            # saver.restore(sess, './model/captcha_model')

            predict = tf.argmax(tf.reshape(output, [-1, captcha, chars_len]), 2)

            for i in range(10):  # 进行10次验证
                t, x = gen_captcha.gen_test_img()
                text_list = sess.run(predict, feed_dict={self.X: [x], self.prob: 1})

                text = text_list[0].tolist()
                vector = np.zeros(captcha * chars_len)
                i = 0
                for n in text:
                    vector[i * chars_len + n] = 1
                    i += 1
                p_text = gen_captcha.vec2text(vector)

                print('验证码为:{}, 预测为:{}'.format(t, p_text))

        return True

    def identification(self, x):
        """接口调用的识别函数"""
        output = self.captcha_cnn()
        saver = tf.train.Saver()

        with tf.Session() as sess:
            saver.restore(sess, './model/captcha_model')
            predict = tf.argmax(tf.reshape(output, [-1, captcha, chars_len]), 2)

            text_list = sess.run(predict, feed_dict={self.X: [x], self.prob: 1})
            text = text_list[0].tolist()
            vector = np.zeros(captcha * chars_len)
            i = 0
            for n in text:
                vector[i * chars_len + n] = 1
                i += 1
            p_text = gen_captcha.vec2text(vector)

        tf.reset_default_graph()
        return p_text


if __name__ == '__main__':
    cv = Captcha_Verify()

    # cv.training_cnn()

    cv.captcha_identification()

    # t, x = gen_captcha.gen_test_img()
    # cv.identification(x)
