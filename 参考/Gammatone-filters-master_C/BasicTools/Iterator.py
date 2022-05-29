import numpy as np


class Iterator:
    def __init__(self, x, shuffle=True):
        """ x: list or ndarray
        """

        if isinstance(x, list):
            n_item = len(x)
        elif isinstance(x, np.ndarray):
            n_item = x.shape[0]
        else:
            raise Exception(f'unsupported type {type(x)}')

        self.x = x
        if shuffle:
            np.random.shuffle(self.x)
        #
        self._n_item = n_item  # number of elements in x
        self._item_pointer = -1   # index of element avaliable new

    def reach_end(self, n_keep=0):
        """ whether there are n_keep item left
        """
        return self._item_pointer == self._n_item-1-n_keep

    def next(self):
        """ get next element, None will return if reach the end
        """
        if self.reach_end():
            item = None
        else:
            self._item_pointer += 1
            item = self.x[self._item_pointer]
        return item

    def go_back(self, n_step):
        """ move _item_pointer backwards
        """
        if n_step > self._item_pointer:
            self._item_pointer = -1
        else:
            self._item_pointer -= n_step

    def reset(self, shuffle=True):
        if shuffle:
            np.random.shuffle(self.x)
        self._item_pointer = -1


if __name__ == '__main__':
    generator = Iterator([1, 2, 3, 4, 5])
    while True:
        value = generator.next()

        print(value)
        if value is None:
            generator.go_back(2)
