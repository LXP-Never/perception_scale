import numpy as np
import time


class ProgressBar(object):
    def __init__(self, max_value=100, is_show_resrc=False, is_show_time=False):

        self.max_value = max_value
        self.value = 0.
        self.is_show_resrc = is_show_resrc
        self.is_show_time = is_show_time
        self.n_line_pre = 0

    def get_cur_value(self):
        print(self.value)

    def is_finish(self):
        return self.value >= self.max_value

    def _get_n_col_bar(self):
        # n_col_terminal = os.get_terminal_size()[0]
        # n_col_bar = int(n_col_terminal*0.66)
        n_col_bar = 50
        return n_col_bar

    def update(self, text=''):
        if self.n_line_pre > 0:
            print(f'\033[{self.n_line_pre+1}A')  # move to start position
            [print('\033[K') for _ in range(self.n_line_pre)]
            print(f'\033[{self.n_line_pre+1}A')  # move to start position

        self.value = self.value + 1
        p = np.float32(self.value)/self.max_value
        n_col_bar = self._get_n_col_bar()
        n_finish = np.int16(p*n_col_bar)
        n_left = n_col_bar - n_finish
        bar_str = f"{n_finish*'>'}{n_left*'='}"
        bar_status_str = f'[{bar_str}] {p*100:>3.0f}% \n {text}'
        print(bar_status_str)
        self.n_line_pre = bar_status_str.count('\n')+1


if __name__ == '__main__':

    pb = ProgressBar(100)
    for i in range(100):
        time.sleep(0.1)
        pb.update(text='test\n'*(i % 5))
