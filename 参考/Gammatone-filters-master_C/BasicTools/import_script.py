import importlib
import numpy as np
import string


def gen_random_name(name_len=4):
    letters = string.ascii_letters
    n_letter = len(letters)
    name = ''.join([letters[np.random.randint(0, n_letter)]
                    for i in range(name_len)])
    return name


def import_script(script_path):
    random_name = gen_random_name()
    spec = importlib.util.spec_from_file_location(random_name, script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if __name__ == '__main__':
    model = import_script('import_script.py')
    print(model)
