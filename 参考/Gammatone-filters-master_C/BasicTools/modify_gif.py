import argparse
from PIL import Image


def modify_gif(gif_path, new_gif_path, duration=None):
    gif = Image.open(gif_path)
    gif.info['duration'] = duration
    if new_gif_path is None:
        new_gif_path = gif_path

    gif.save(new_gif_path, save_all=True)


def parse_args():
    parser = argparse.ArgumentParser(description='parse argments')
    parser.add_argument(
        '--gif-path', dest='gif_path', required=True, type=str,
        help='path of the input file')
    parser.add_argument(
        '--new-gif-path', dest='new_gif_path', required=True, type=str,
        help='')
    parser.add_argument(
        '--duration', dest='duration', type=int, default=None,
        help='new duration')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    modify_gif(args.gif_path, args.new_gif_path, args.duration)
