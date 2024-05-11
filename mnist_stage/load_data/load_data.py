from __future__ import absolute_import, division, print_function, \
    unicode_literals
from pathlib import Path

import argparse
import numpy as np


# load data
def load_data(path):
    with np.load(path) as f:
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']

    return (x_train, y_train), (x_test, y_test)


# do data transform
def transform(output_dir, file_name, x_train_path_file, x_test_path_file, y_train_path_file, y_test_path_file):
    (x_train, y_train), (x_test, y_test) = load_data(output_dir + file_name)
    print("### loading data done.")

    # save data
    x_train, x_test = x_train / 255.0, x_test / 255.0

    x_train_path = output_dir + "x_train.npy"
    x_test_path = output_dir + "x_test.npy"
    y_train_path = output_dir + "y_train.npy"
    y_test_path = output_dir + "y_test.npy"

    Path(x_train_path_file).parent.mkdir(parents=True, exist_ok=True)
    Path(x_test_path_file).parent.mkdir(parents=True, exist_ok=True)
    Path(y_train_path_file).parent.mkdir(parents=True, exist_ok=True)
    Path(y_test_path_file).parent.mkdir(parents=True, exist_ok=True)
    np.save(x_train_path, x_train)
    np.save(x_test_path, x_test)
    np.save(y_train_path, y_train)
    np.save(y_test_path, y_test)

    Path(x_train_path_file).write_text(x_train_path)
    Path(x_test_path_file).write_text(x_test_path)
    Path(y_train_path_file).write_text(y_train_path)
    Path(y_test_path_file).write_text(y_test_path)
    print("### data transform done.")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Kubeflow MNIST load data script')
    parser.add_argument('--data_dir', type=str, required=True, help='train feature dataset path.')
    parser.add_argument('--file_name', type=str, required=True, help='local file to be input')
    parser.add_argument('--x_train_path_file', type=str, required=True, help='file saved x_train data path')
    parser.add_argument('--x_test_path_file', type=str, required=True, help='file saved x_test data path')
    parser.add_argument('--y_train_path_file', type=str, required=True, help='file saved y_train data path')
    parser.add_argument('--y_test_path_file', type=str, required=True, help='file saved y_test data path')
    args = parser.parse_args()
    return args


def run():
    args = parse_arguments()
    transform(args.data_dir, args.file_name, args.x_train_path_file, args.x_test_path_file, args.y_train_path_file,
              args.y_test_path_file)


if __name__ == '__main__':
    run()
