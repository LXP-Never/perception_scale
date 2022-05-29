import numpy as np


def make_onehot(y, n_class, class_labels=None):
    if isinstance(y, list) and isinstance(y[0], str):
        # y is a list of string
        if class_labels is not None:
            label_id_dict = {str(label): i
                             for i, label in enumerate(class_labels)}
        else:
            label_id_dict = {str(i): i for i in range(n_class)}
        n_sample = len(y)
        onehot = np.zeros((n_sample, n_class), dtype=np.float32)
        for i, label in enumerate(y):
            onehot[i, label_id_dict[label]] = 1
    else:
        # y is list of int or np.ndarray
        y = np.squeeze(np.asarray(y, dtype=np.int))
        n_sample = y.shape[0]
        onehot = np.zeros((n_sample, n_class), dtype=np.float32)
        for i in range(n_sample):
            onehot[i, y[i]] = 1
    return onehot


if __name__ == '__main__':
    y = ['a', 'b', 'e', 'd']
    onehot = make_onehot(y, 8, ['a', 'b', 'c', 'd', 'e'])
    print(np.argmax(onehot, axis=1))

    y = [1, 2, 3, 4, 5]
    onehot = make_onehot(y, 8)
    print(np.argmax(onehot, axis=1))
