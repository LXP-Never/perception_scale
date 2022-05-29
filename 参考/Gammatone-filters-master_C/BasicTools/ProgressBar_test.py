from BasicTools import ProcessBarMulti
from multiprocessing import Process
import time


def count_up(pb_share, token, max_value):
    for value in range(max_value):
        # with lock:
        pb_share.update(token)
        time.sleep(0.1)


def ProcessBarMulti_test():
    # import queue

    n_task = 3
    process_all = []
    max_value_all = [10, 20, 15]
    pb_share = ProcessBarMulti(max_value_all[:n_task])
    for task_i in range(n_task):
        process = Process(target=count_up,
                          args=(pb_share, str(task_i),
                                max_value_all[task_i]))
        process.start()
        process_all.append(process)
    [process.join() for process in process_all]


if __name__ == '__main__':
    ProcessBarMulti_test()
