import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import fashion_mnist
from sklearn.model_selection import train_test_split
import random

# Load and preprocess data
def load_and_preprocess_data():
    (x_train_val, y_train_val), (x_test, y_test) = fashion_mnist.load_data()
    x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, test_size=0.2, random_state=42)
    x_train = x_train / 255.0
    x_val = x_val / 255.0
    x_test = x_test / 255.0
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)

# Create pairs
def create_pairs(images, labels):
    pair_images, pair_labels = [], []
    n = min([len(labels[l]) for l in range(10)]) - 1
    for l in range(10):
        for i in range(n):
            z1, z2 = labels[l][i], labels[l][i + 1]
            pair_images += [[images[z1], images[z2]]]
            inc = random.randrange(1, 10)
            dn = (l + inc) % 10
            z1, z2 = labels[l][i], labels[dn][i]
            pair_images += [[images[z1], images[z2]]]
            pair_labels += [1, 0]
    return np.array(pair_images), np.array(pair_labels)

def prepare_data():
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_and_preprocess_data()

    digit_indices = [np.where(y_train == i)[0] for i in range(10)]
    train_image_pairs, train_label_pairs = create_pairs(x_train, digit_indices)

    digit_indices = [np.where(y_val == i)[0] for i in range(10)]
    val_image_pairs, val_label_pairs = create_pairs(x_val, digit_indices)

    digit_indices = [np.where(y_test == i)[0] for i in range(10)]
    test_image_pairs, test_label_pairs = create_pairs(x_test, digit_indices)

    return (train_image_pairs, train_label_pairs), (val_image_pairs, val_label_pairs), (test_image_pairs, test_label_pairs)

# Define model
class BaseNetwork(models.Model):
    def __init__(self):
        super(BaseNetwork, self).__init__()
        self.flatten = layers.Flatten()
        self.d1 = layers.Dropout(0.2)
        self.fc1 = layers.Dense(256, activation='relu')
        self.d2 = layers.Dropout(0.2)
        self.fc2 = layers.Dense(128, activation='relu')

    def call(self, inputs):
        x = self.flatten(inputs)
        x = self.d1(x)
        x = self.fc1(x)
        x = self.d2(x)
        return self.fc2(x)

def create_model(input_shape):
    base_network = BaseNetwork()
    left_input = layers.Input(shape=input_shape)
    right_input = layers.Input(shape=input_shape)
    left_output = base_network(left_input)
    right_output = base_network(right_input)

    def euclidean(vect):
        x, y = vect
        sum_square = tf.keras.backend.sum(tf.keras.backend.square(x - y), axis=-1, keepdims=True)
        return tf.keras.backend.sqrt(tf.keras.backend.maximum(sum_square, tf.keras.backend.epsilon()))

    output = layers.Lambda(euclidean)([left_output, right_output])
    model = models.Model([left_input, right_input], output)
    return model

def contrastive_loss_with_margin(margin):
    def contrastive_loss(y_true, y_pred):
        squared_pred = tf.keras.backend.square(y_pred)
        squared_margin = tf.keras.backend.square(tf.keras.backend.maximum(margin - y_pred, 0))
        return y_true * squared_pred + (1 - y_true) * squared_margin
    return contrastive_loss

def train_model():
    (train_image_pairs, train_label_pairs), (val_image_pairs, val_label_pairs), (test_image_pairs, test_label_pairs) = prepare_data()

    input_shape = train_image_pairs[0, 0].shape
    model = create_model(input_shape)

    optimizer = tf.keras.optimizers.RMSprop()
    loss = contrastive_loss_with_margin(margin=1)
    model.compile(loss=loss, optimizer=optimizer)
    model.summary()

    model.fit(
        [train_image_pairs[:, 0], train_image_pairs[:, 1]],
        train_label_pairs,
        epochs=50,
        batch_size=256,
        validation_data=([val_image_pairs[:, 0], val_image_pairs[:, 1]], val_label_pairs)
    )

    def compute_accuracy(y_true, y_pred):
        yhat = y_pred.ravel() < 0.5
        return np.mean(yhat == y_true)

    y_pred_test = model.predict([test_image_pairs[:, 0], test_image_pairs[:, 1]])
    test_accuracy = compute_accuracy(test_label_pairs, y_pred_test)
    print("Test accuracy: {:.2f}%".format(test_accuracy * 100))

    model.save('model/fashion_recommendation_model.keras')

if __name__ == '__main__':
    train_model()
