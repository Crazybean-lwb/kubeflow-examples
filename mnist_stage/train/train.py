from __future__ import absolute_import, division, print_function, \
    unicode_literals

import argparse
from datetime import datetime
import os
from pathlib import Path

import numpy as np
import tensorflow as tf


# train model
def train_model(x_train_path, x_test_path, y_train_path, y_test_path, output_model_path_file):
    x_train_data = np.load(x_train_path)
    x_test_data = np.load(x_test_path)
    y_train_data = np.load(y_train_path)
    y_test_data = np.load(y_test_path)

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

    model.fit(x_train_data, y_train_data, batch_size=32, epochs=5, callbacks=callbacks,
              validation_data=(x_test_data, y_test_data))

    model_path = os.path.dirname(x_train_path) + '/mnist_model.h5'
    model.save(model_path)
    Path(output_model_path_file).parent.mkdir(parents=True, exist_ok=True)
    Path(output_model_path_file).write_text(model_path)
    print("### save model done.")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Kubeflow MNIST train model script')
    parser.add_argument('--x_train_path', type=str, required=True, help='train feature dataset path.')
    parser.add_argument('--x_test_path', type=str, required=True, help='test feature dataset path.')
    parser.add_argument('--y_train_path', type=str, required=True, help='train label dataset path.')
    parser.add_argument('--y_test_path', type=str, required=True, help='test label dataset path.')
    parser.add_argument('--output_model_path_file', type=str, required=True, help='trained model path.')
    args = parser.parse_args()
    return args


def run():
    args = parse_arguments()
    train_model(args.x_train_path, args.x_test_path, args.y_train_path, args.y_test_path, args.output_model_path_file)


if __name__ == '__main__':
    run()
