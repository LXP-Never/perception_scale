import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf  # noqa: E02


def select_gpu(gpu_id):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_visible_devices(gpus[gpu_id], 'GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
