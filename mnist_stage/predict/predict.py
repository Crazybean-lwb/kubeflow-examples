from __future__ import absolute_import, division, print_function, \
    unicode_literals

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf


def predict(model_path, x_test_path, y_test_path, result_path_file):
    result_path = os.path.dirname(x_test_path) + '/result.csv'
    model = tf.keras.models.load_model(model_path)

    x_test_data = np.load(x_test_path)
    y_test_data = np.load(y_test_path)

    pre = model.predict(x_test_data)
    model.evaluate(x_test_data, y_test_data)
    df = pd.DataFrame(data=pre,
                      columns=["prob_0", "prob_1", "prob_2", "prob_3", "prob_4", "prob_5", "prob_6", "prob_7", "prob_8",
                               "prob_9"])
    y_real = pd.DataFrame(data=y_test_data, columns=["real_number"])
    result = pd.concat([df, y_real], axis=1)
    result.to_csv(result_path)
    Path(result_path_file).parent.mkdir(parents=True, exist_ok=True)
    Path(result_path_file).write_text(result_path)
    print("### save predict result file: result.csv")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Kubeflow MNIST predict model script')
    parser.add_argument('--model_path', type=str, required=True, help='trained model path.')
    parser.add_argument('--x_test_path', type=str, required=True, help='test feature dataset path.')
    parser.add_argument('--y_test_path', type=str, required=True, help='test label dataset path.')
    parser.add_argument('--result_path_file', type=str, required=True, help='file saved result path.')
    args = parser.parse_args()
    return args


def run():
    args = parse_arguments()
    predict(args.model_path, args.x_test_path, args.y_test_path, args.result_path_file)


if __name__ == '__main__':
    run()