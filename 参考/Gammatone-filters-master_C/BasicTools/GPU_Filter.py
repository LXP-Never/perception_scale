import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
import tensorflow as tf  #noqa:E402


class GPU_Filter:
    def __init__(self, gpu_id=0):
        gpus = tf.config.experimental.list_physical_devices('GPU')
        try:
            tf.config.experimental.set_visible_devices(gpus[gpu_id], 'GPU')
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            None

    def filter(self, x, coef):
        # ensure all data is float32
        x = x.astype(np.float32)
        coef = coef.astype(np.float32)

        if len(x.shape) > 2:
            raise Exception('only 1d array is supported')
        coef_conv = np.concatenate((np.flip(coef),
                                    np.zeros(coef.shape[0], dtype=np.float32)))
        y = tf.nn.conv1d(x[np.newaxis, :, np.newaxis],
                         coef_conv[:, np.newaxis, np.newaxis],
                         stride=1,
                         padding='SAME')
        return np.asarray(tf.squeeze(y))

    def brir_filter(self, x, brir):
        if brir is None:
            return x.copy()
        y_l = self.filter(x, brir[:, 0])
        y_r = self.filter(x, brir[:, 1])
        return np.asarray((y_l, y_r)).T
