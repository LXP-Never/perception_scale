import os
import numpy as np
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class GPU_Filter:
    def __init__(self, gpu_index):
        self._graph = tf.Graph()
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.visible_device_list = '{}'.format(gpu_index)
        self._sess = tf.compat.v1.Session(graph=self._graph, config=config)
        self._build_model()

    def _build_model(self):
        with self._graph.as_default():
            x = tf.compat.v1.placeholder(dtype=tf.float64,
                                         shape=(None, None, 1))
            coef = tf.compat.v1.placeholder(dtype=tf.float64,
                                            shape=(None, 1, 1))
            coef_flip_pad = tf.pad(tf.reverse(coef, axis=[0]),
                                   paddings=[[0, tf.shape(coef)[0]-1],
                                             [0, 0], [0, 0]])
            y = tf.nn.convolution(input=x, filter=coef_flip_pad,
                                  padding='SAME')

            init = tf.compat.v1.global_variables_initializer()
            self._sess.run(init)

            self._x = x
            self._coef = coef
            self._y = y

    def filter(self, x, coef):
        x_shape = x.shape
        if len(x_shape) == 1:
            x.shape = [1, x_shape[0], 1]
        y = self._sess.run(self._y, feed_dict={self._x: x,
                                               self._coef: coef[:, np.newaxis,
                                                                np.newaxis]})
        x.shape = x_shape
        return np.squeeze(y)

    def brir_filter(self, x, brir):
        if brir is None:
            return x.copy()
        y_l = self.filter(x, brir[:, 0])
        y_r = self.filter(x, brir[:, 1])
        return np.asarray((y_l, y_r)).T
