import os


def get_realpath(file_path):
    """ if file_path is a link, the real path of this link rather than the
    file the link refers is returned
    """
    if os.path.islink(file_path):
        file_name = os.path.basename(file_path)
        dir_path = os.path.dirname(file_path)
        realpath = file_name
        while os.path.islink(dir_path):
            dir_name = os.path.basename(dir_path)
            realpath = f'{dir_name}/{realpath}'
            dir_path = os.path.dirname(dir_path)
        dir_path = os.path.realpath(dir_path)
        realpath = f'{dir_path}/{realpath}'
    else:
        realpath = os.path.realpath(file_path)
    return realpath


def get_path_depth(path):
    """ count how many levels in given path
    """
    if path == '.':  # current directory
        depth = 0
    else:
        path = path.strip('/')
        dir_names = [item for item in path.split('/') if len(item) > 0]
        if '..' in dir_names:
            raise Exception('.. occurs in path')
        depth = len(dir_names)
    return depth


def file_path_filter(file_path, suffix, filter_func):
    is_keep = False
    if suffix is None and filter_func is None:
        is_keep = True
    elif suffix is not None:
        is_keep = \
            os.path.basename(file_path).split('.')[-1] == suffix.split('.')[-1]
        if is_keep and filter_func is not None:
            is_keep = filter_func(file_path)
    return is_keep


def get_file_path(dir_path, suffix=None, filter_func=None, is_absolute=False,
                  max_depth=-1):
    """ return a list of file paths
    """

    # os.walk is very slow, if max_depth=0, use os.listdir instead
    if max_depth == 0:
        file_names = os.listdir(dir_path)
        file_paths_raw = [f'{dir_path}/{file_name}'
                          for file_name in file_names]
    else:
        file_paths_raw = []
        for file_dir_path, _, file_names in os.walk(dir_path):
            depth = get_path_depth(os.path.relpath(file_dir_path, dir_path))
            if max_depth > 0 and depth > max_depth:  #
                continue
            file_paths_raw.extend(
                [f'{file_dir_path}/{file_name}' for file_name in file_names])

    # apply path filter
    file_paths = [item.replace('//', '/') for item in file_paths_raw
                  if file_path_filter(item, suffix, filter_func)]
    if is_absolute:
        file_paths = [get_realpath(item) for item in file_paths]
    else:
        file_paths = [os.path.relpath(item, dir_path)
                      for item in file_paths]

    return file_paths


if __name__ == '__main__':
    import sys
    dir_path, max_depth = sys.argv[1:3]
    max_depth = int(max_depth)
    file_path_all = get_file_path(dir_path, max_depth=max_depth)
    print(f'files and directories under {dir_path}, max_depth is {max_depth}')
    print(file_path_all)
