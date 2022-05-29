import os
import re
import numpy as np
import subprocess
from subprocess import PIPE
from multiprocessing import Queue, Process, Manager, Lock
from BasicTools.ProgressBarMulti import ProgressBarMulti


def get_subprocess_pid(pid):
    p = subprocess.Popen(['pstree', '-pn', f'{pid}'], stdout=PIPE)
    out, err = p.communicate()
    pids = []
    for line in out.split():
        elems = re.split('[()]', line.decode().strip())
        if len(elems) > 2:
            try:
                pid = int(elems[-2])
            except Exception:
                None
        pids.append(pid)
    return pids


def worker(func, lock, tasks_queue, outputs_dict, pb, worker_params,
           father_pid, worker_base_dir):
    cur_pid = os.getpid()
    worker_dir = f'{worker_base_dir}/{cur_pid}'
    if os.path.exists(worker_dir):
        os.system(f'rm -r {worker_dir}')
    os.makedirs(worker_dir)
    while True:
        task = tasks_queue.get()
        if task is None:
            break
        else:
            task_id = task[0]
            try:
                n_param = len(worker_params)
                for param_i in range(n_param):
                    if worker_params[param_i] == 'pid':
                        worker_params[param_i] = cur_pid
                result = func(*task[1:], *worker_params)
            except Exception as e:
                with open(f'{worker_dir}/log', 'w') as except_file:
                    except_file.write(f'{e}')
                print(f'{e}')
                print(f'log info in {worker_dir}')
                print(f'running {func}: kill all subprocess')

                for pid_str in os.listdir(f'{worker_base_dir}'):
                    if pid_str.isdecimal():
                        pid = int(pid_str)
                        if pid != cur_pid:
                            os.system(f'kill {pid} > /dev/null')
                os.system(f'kill {father_pid} > /dev/null')

                # rerun to show more information absout exception
                result = func(*task[1:], *worker_params)
                return None

            with lock:
                outputs_dict[task_id] = result
                if pb is not None:
                    pb.update()
    os.system(f'rm -r {worker_dir}')
    return None


def easy_parallel(func, tasks, n_worker=8, show_progress=False,
                  worker_params=None, dump_dir='dump'):
    """
    Args:
        func: function to be called in parallel
        tasks: list of list or 2 dimension ndarray, arguments of func,
        n_worker: number of processes
        show_progress: show progress bar
        worker_params: params to each worker
    """

    if len(tasks) < 1:
        return None

    if isinstance(tasks, np.ndarray):
        tasks = tasks.tolist()

    # # avoid n_worker is too large for current machine
    # n_worker_max = len(list(os.sched_getaffinity(0)))
    # n_worker = np.min((n_worker, n_worker_max))

    threads = []
    outputs_dict = Manager().dict()
    if show_progress:
        pb = ProgressBarMulti([len(tasks)])
    else:
        pb = None

    tasks_queue = Queue()
    [tasks_queue.put([str(task_i), *task])
     for task_i, task in enumerate(tasks)]
    [tasks_queue.put(None) for worker_i in range(n_worker)]

    cur_pid = os.getpid()
    rand_generator = np.random.RandomState(cur_pid)
    while True:
        rand_num = rand_generator.randint(0, 10000)
        worker_base_dir = f'{dump_dir}/easy_parallel_{rand_num}'
        if not os.path.exists(worker_base_dir):
            break
        os.makedirs(worker_base_dir)

    if worker_params is None:
        worker_params = [[] for worker_i in range(n_worker)]
    else:
        n_worker = len(worker_params)

    for worker_i in range(n_worker):
        n_param = len(worker_params[worker_i])
        for param_i in range(n_param):
            if worker_params[worker_i][param_i] == 'randomstate':
                rand_seed = int(time.time())+worker_i
                worker_params[worker_i][param_i] = \
                    np.random.RandomState(rand_seed)

    lock = Lock()
    for worker_i in range(n_worker):
        thread = Process(
                target=worker,
                args=(func, lock, tasks_queue, outputs_dict, pb,
                      worker_params[worker_i], cur_pid, worker_base_dir))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    with open(f'{worker_base_dir}/result_keys.txt', 'w') as key_record_file:
        key_record_file.write('; '.join(outputs_dict.keys()))
        key_record_file.write(f'\n n_task {len(tasks)}')

    outputs = [outputs_dict[str(task_i)] for task_i, _ in enumerate(tasks)]

    os.system(f'rm -r {worker_base_dir}')
    return outputs


if __name__ == '__main__':
    import time

    def test_func(*args):
        print(args)
        time.sleep(np.random.randint(10, size=1))
        # raise Exception(f'{args}')
        return args

    # tasks = [[i] for i in range(32)]
    tasks = np.random.rand(32, 2)
    outputs = easy_parallel(test_func, tasks, show_process=True)
    print(outputs)
