import sys
import json
import argparse

import tensorflow as tf
from tensorflow.keras.models import load_model, save_model
from sklearn.metrics import confusion_matrix

import numpy as np

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)


def load_dataset():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    return x_train, x_test, y_train, y_test


def get_model():
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10),
        ]
    )
    return model


def train(model_filename, epochs):
    model = get_model()
    x_train, x_test, y_train, y_test = load_dataset()
    model.compile(
        optimizer="adam", loss=loss_fn, metrics=["sparse_categorical_accuracy"]
    )
    tbCallBack = tf.keras.callbacks.TensorBoard(log_dir="/log_dir", write_images=True)
    model.fit(x_train, y_train, epochs=epochs, callbacks=None)
    model.save(model_filename)
    #metadata = {
    #    'outputs' : [{
    #        'type': 'tensorboard',
    #        'source': "/log_dir"
    #    }]
    #}
    #with open('/mlpipeline-ui-metadata.json', 'w') as f:
    #    json.dump(metadata, f)
    result = model.evaluate(x_test, y_test, verbose=2)
    print(result)


def test(model_filename):
    model = load_model(model_filename)
    model.summary()
    x_train, x_test, y_train, y_test = load_dataset()
    result = model.evaluate(x_test, y_test, verbose=2)
    y_predicted = model.predict(x_test)
    conf_mat = confusion_matrix(
        y_test, np.argmax(y_predicted, axis=1), labels=range(10)
    )

    np.savetxt("/conf_mat.csv", conf_mat, '%s', delimiter=",")
    metadata = {
        'outputs' : [{
            'type': 'confusion_matrix',
            'format': 'csv',
            'schema': [
            {'name': 'target', 'type': 'CATEGORY'},
            {'name': 'predicted', 'type': 'CATEGORY'},
            {'name': 'count', 'type': 'NUMBER'},
            ],
            'source': '/conf_mat.csv',
            'labels': list(map(str, range(10))),
        }]
    }
    with open('/mlpipeline-ui-metadata.json', 'w') as f:
        json.dump(metadata, f)

    print(conf_mat)
    print(result)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--train", type=str, default=None)
    parser.add_argument("--test", type=str, default=None)
    args = parser.parse_args()

    if not args.train is None:
        train(args.train, args.epochs)
    elif not args.test is None:
        test(args.test)
    else:
        raise ValueError("choose either train or test")
