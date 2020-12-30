from __future__ import absolute_import, division, print_function, \
    unicode_literals

import argparse
from datetime import datetime
import numpy as np
import tensorflow as tf


# train model
def train_model(data_file, output_dir):
    """
    all file use absolute dir
    :param data_file: `train_test_data.txt` absolute dir
    :param output_dir:
    :return:
    """
    model_file = 'mnist_model.h5'

    data_list = data_file.split(',')
    with open(data_list[0], 'rb') as f:
        x_train = np.load(f)
    with open(data_list[1], 'rb') as f:
        x_test = np.load(f)
    with open(data_list[2], 'rb') as f:
        y_train = np.load(f)
    with open(data_list[3], 'rb') as f:
        y_test = np.load(f)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    print(model.summary())

    callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir=datetime.now().date().__str__()),
        # Interrupt training if `val_loss` stops improving for over 2 epochs
        tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
    ]

    model.fit(x_train, y_train, batch_size=32, epochs=5, callbacks=callbacks,
              validation_data=(x_test, y_test))

    model.save(output_dir + model_file)
    print("### save model done.")

    with open(output_dir + 'model.txt', 'w') as f:
        f.write(output_dir + model_file)
    print("### write trained model path and name to: model.txt done.")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Kubeflow MNIST train model script')
    parser.add_argument('--data_dir', type=str, required=True, help='local file dir')
    parser.add_argument('--data_file', type=str, required=True, help='a file write train and test data absolute dir')
    args = parser.parse_args()
    return args


def run():
    args = parse_arguments()
    train_model(args.data_file, args.data_dir)


if __name__ == '__main__':
    run()
