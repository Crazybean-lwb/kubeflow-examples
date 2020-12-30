from __future__ import absolute_import, division, print_function, \
    unicode_literals

import argparse
import numpy as np
import pandas as pd
import tensorflow as tf


def predict(output_dir, model_file, data_file):
    """
    all file use absolute dir
    :param output_dir:
    :param model_file: `model.txt` absolute dir
    :param data_file: `train_test_data.txt` absolute dir
    :return:
    """
    model = tf.keras.models.load_model(model_file)

    data_list = data_file.split(',')
    with open(data_list[1], 'rb') as f:
        x_test = np.load(f)
    with open(data_list[3], 'rb') as f:
        y_test = np.load(f)

    pre = model.predict(x_test)
    model.evaluate(x_test, y_test)
    df = pd.DataFrame(data=pre,
                      columns=["prob_0", "prob_1", "prob_2", "prob_3", "prob_4", "prob_5", "prob_6", "prob_7", "prob_8",
                               "prob_9"])
    y_real = pd.DataFrame(data=y_test, columns=["real_number"])
    result = pd.concat([df, y_real], axis=1)
    result.to_csv(output_dir + 'result.csv')
    print("### save predict result file: result.csv")

    with open(output_dir + 'result.txt', 'w') as f:
        f.write(output_dir + 'result.csv')
    print("### write result path and name to: result.txt done.")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Kubeflow MNIST predict model script')
    parser.add_argument('--data_dir', type=str, required=True, help='local file dir')
    parser.add_argument('--model_file', type=str, required=True, help='a file write trained model absolute dir')
    parser.add_argument('--data_file', type=str, required=True, help='la file write train and test data absolute dir')
    args = parser.parse_args()
    return args


def run():
    args = parse_arguments()
    predict(args.data_dir, args.model_file, args.data_file)


if __name__ == '__main__':
    run()
