import tensorflow as tf
import sys
from tensorflow.keras.models import load_model, save_model

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

def load_dataset():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    return x_train, x_test, y_train, y_test

def get_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10)
    ])
    return model

def train(model_filename):
    model = get_model()
    x_train, x_test, y_train, y_test = load_dataset()
    model.compile(optimizer='adam', loss=loss_fn, metrics=['sparse_categorical_accuracy'])
    model.fit(x_train, y_train, epochs=5)
    model.save(model_filename)
    result = model.evaluate(x_test, y_test, verbose=2)
    print(result)

def test(model_filename):
    model = load_model(model_filename)
    model.summary()
    x_train, x_test, y_train, y_test = load_dataset()
    result = model.evaluate(x_test, y_test, verbose=2)
    print(result)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(1)
    if sys.argv[1] == "--train":
        train(sys.argv[2])
    elif sys.argv[1] == "--test":
        test(sys.argv[2])
    else:
        raise ValueError("choose either train or test")
