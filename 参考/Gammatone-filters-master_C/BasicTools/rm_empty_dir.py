import os
import argparse


def is_dir_empty(dir_path):
    contents = os.listdir(dir_path)
    return len(contents) == 0


def rm_empty_dir(dir_path):

    is_done = False
    while not is_done:
        is_done = True
        for root, sub_dir_names, _ in os.walk(dir_path):
            for sub_dir_name in sub_dir_names:
                sub_dir_path = f'{root}/{sub_dir_name}'
                if is_dir_empty(sub_dir_path):
                    os.rmdir(sub_dir_path)
                    print(sub_dir_path)
                    is_done = False
            if not is_done:
                break
    if is_dir_empty(dir_path):
        os.rmdir(dir_path)
        print(dir_path)


def parse_args():
    parser = argparse.ArgumentParser(description='remove empty directories \
                                     recursively')
    parser.add_argument('--dir', dest='dir_path', required=True, type=str,
                        help='')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    rm_empty_dir(args.dir_path)


if __name__ == '__main__':
    main()
