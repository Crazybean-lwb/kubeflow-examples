from __future__ import absolute_import, division, print_function, \
    unicode_literals

import argparse
import numpy as np


# load data
def load_data(path):
    with np.load(path) as f:
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']

    return (x_train, y_train), (x_test, y_test)


# do data transform
def transform(output_dir, file_name):
    x_train_name = 'x_train.npy'
    x_test_name = 'x_test.npy'
    y_train_name = 'y_train.npy'
    y_test_name = 'y_test.npy'
    (x_train, y_train), (x_test, y_test) = load_data(output_dir + file_name)
    print("### loading data done.")

    x_train, x_test = x_train / 255.0, x_test / 255.0
    np.save(output_dir + x_train_name, x_train)
    np.save(output_dir + x_test_name, x_test)
    np.save(output_dir + y_train_name, y_train)
    np.save(output_dir + y_test_name, y_test)
    print("### data transform done.")

    with open(output_dir + 'train_test_data.txt', 'w') as f:
        f.write(output_dir + x_train_name + ',')
        f.write(output_dir + x_test_name + ',')
        f.write(output_dir + y_train_name + ',')
        f.write(output_dir + y_test_name)
    print("### write train and test data name to: train_test_data.txt done.")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Kubeflow MNIST load data script')
    parser.add_argument('--data_dir', type=str, required=True, help='local file dir')
    parser.add_argument('--file_name', type=str, required=True, help='local file to be input')
    args = parser.parse_args()
    return args


def run():
    args = parse_arguments()
    transform(args.data_dir, args.file_name)


if __name__ == '__main__':
    run()
