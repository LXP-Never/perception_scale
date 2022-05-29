from multiprocessing import Process


def run_in_back(func, *args_list):
    thread = Process(target=func, args=args_list, daemon=False)
    thread.start()
    return


if __name__ == '__main__':
    import time

    def test_func(x):
        for i in range(5):
            print(i)
            time.sleep(2)
    run_in_back(test_func, 10)
    print('finish')
