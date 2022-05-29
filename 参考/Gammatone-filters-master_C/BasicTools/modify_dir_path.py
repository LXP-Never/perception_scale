import os
import shutil
import argparse


def expand_dirs(dir_paths):
    sub_dir_paths = []
    for dir_path in dir_paths:
        entry_names = os.listdir(dir_path)
        for entry_name in entry_names:
            entry_path = f'{dir_path}/{entry_name}'
            if os.path.isdir(entry_path):
                sub_dir_paths.append(entry_path)
    return sub_dir_paths


def modify_path(dir_path, src_pattern, dest_pattern, max_try=-1):
    #
    src_pattern = src_pattern.strip('/')
    src_pattern_len = len(src_pattern)
    dest_pattern = dest_pattern.strip('/')

    sub_dir_paths = [dir_path]
    n_try = 0
    while True:
        sub_dir_paths = expand_dirs(sub_dir_paths)
        if len(sub_dir_paths) == 0:  # reach the bottom
            break
        unmatched_sub_dir_paths = []
        for sub_dir_path in sub_dir_paths:
            if sub_dir_path[-src_pattern_len:] == src_pattern:
                src_path = sub_dir_path
                dest_path = f'{sub_dir_path[:-src_pattern_len]}/{dest_pattern}'
                if os.path.exists(dest_path):
                    raise Exception(f'{dest_path} alread exists')
                #
                is_continue = input(f'move {src_path} to ${dest_path} ? y/n ')
                if is_continue == '' or is_continue == 'y':
                    shutil.move(src_path, dest_path)
                    n_try = n_try+1
                    if max_try > 0 and n_try >= max_try:
                        break
            else:
                unmatched_sub_dir_paths.append(sub_dir_path)
        sub_dir_paths = unmatched_sub_dir_paths


def parse_args():
    parser = argparse.ArgumentParser(description='parse argments')
    parser.add_argument('--dir', dest='dir_path', required=True, type=str,
                        help='')
    parser.add_argument('--src-pattern', dest='src_pattern', required=True,
                        type=str, help='')
    parser.add_argument('--dest-pattern', dest='dest_pattern', required=True,
                        type=str, help='')
    parser.add_argument('--max-try', dest='max_try', type=int, default=-1,
                        help='specify the maximum number of modification, \
                        default to -1')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    modify_path(dir_path=args.dir_path,
                src_pattern=args.src_pattern,
                dest_pattern=args.dest_pattern,
                max_try=args.max_try)


if __name__ == '__main__':
    main()
