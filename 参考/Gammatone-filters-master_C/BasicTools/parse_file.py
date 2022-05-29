import os
import numpy as np


def file2dict(file_path, numeric=False, squeeze=False, repeat_processor=None):
    """parse file to dictionary
    the content of file should in the following format
    key: value   # value should be in the format: item0; item1, ...
    ....
    Args:
        file_path:
        numeric: bool, whether items are array of numbers.
            if True, value will be parsed to numpy 2d array
        repeat_processor: how to deal with repeat keys
        choices ['keep', 'except', 'none']  'average not implemented
    Returns:
        a dict object
    """
    file_path = os.path.expanduser(file_path)  #

    dict_obj = {}
    with open(file_path, 'r') as dict_file:
        lines = dict_file.readlines()
        for line_i, line in enumerate(lines):
            try:
                line = line.strip()
                if len(line) < 1 or line.startswith('#'):
                    continue

                key, value = line.split(':')
                key = key.strip()
                value = value.strip()
                if numeric:
                    value = np.asarray(
                        [[np.float32(item) for item in row.split()]
                         for row in value.split(';')])
                    if squeeze:
                        value = np.squeeze(value)

                if key in dict_obj.keys():
                    if repeat_processor == 'keep':
                        dict_obj[key].append(value)
                    elif repeat_processor == 'except':
                        raise Exception(f'duplicate keys in {file_path}')
                    else:
                        dict_obj[key] = value
                else:
                    if repeat_processor == 'keep':
                        dict_obj[key] = [value]
                    else:
                        dict_obj[key] = value
            except Exception as e:
                print(f'error in {file_path} line:{line_i}')
                raise Exception(e)
    return dict_obj


def iterable(obj):
    try:
        iter(obj)
    except Exception:
        return False
    else:
        return True


def dict2file(dict_obj, file_path, item_format='', is_sort=True):
    file_path = os.path.expanduser(file_path)

    keys = list(dict_obj.keys())
    if is_sort:
        keys.sort()

    with open(file_path, 'x') as dict_file:
        for key in keys:
            if isinstance(dict_obj[key], str):
                value_str = dict_obj[key]
            elif iterable(dict_obj[key]):
                if iterable(dict_obj[key][0]):
                    # 2 dimension
                    value_str = '; '.join([
                        ' '.join(
                            map(lambda x: ('{:'+item_format+'}').format(x),
                                row))
                        for row in dict_obj[key]])
                else:
                    # 1 dimension
                    value_str = '; '.join(
                        map(lambda x: ('{:'+item_format+'}').format(x),
                            dict_obj[key]))

            else:
                value_str = f'{dict_obj[key]}'
            dict_file.write(f'{key}: {value_str}\n')
