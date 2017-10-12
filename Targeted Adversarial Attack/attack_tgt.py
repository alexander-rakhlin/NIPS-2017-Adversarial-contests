# --input_dir=/home/torch/cleverhans/toolkit/dataset/images --output_dir=output --max_epsilon=16

from os.path import join
import numpy as np
from os import listdir
import argparse
import pandas as pd
from time import time

from custom_nets import InceptionV3, preprocess_input
from keras.layers import Input, Dense, Reshape, Lambda, Add, Activation
from keras.models import Model
import keras.backend as K
from keras.constraints import Constraint
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.callbacks import LambdaCallback, Callback
from PIL import Image
import json
import tensorflow as tf


DATA_SIZE = 120
BATCH_SIZE = 12
GOOD_ACC = 0.96
LR = 0.2
MIRROR = True

TARGET_TIME = 4  # sec/image
MIN_EPOCH = 1
MAX_EPOCH = 5
WARM_UP = 5     # num epochs before start calibrating

INCEPTION_PATH = "inception_v3_weights_tf_dim_ordering_tf_kernels.h5"
SIDE = 299
SHAPE = (3, SIDE, SIDE) if K.image_data_format() == "channels_first" else (SIDE, SIDE, 3)

VERBOSE = True


def load_target_class(input_dir):
    """Loads target classes."""
    df = pd.read_csv(join(input_dir, "target_class.csv"), header=None, index_col=0, names=["Target"])
    return df


def show_weights(w):
    a = np.clip((w.reshape(SHAPE) / 2 + 0.5) * 255, 0, 255)
    img = Image.fromarray(np.array(a, dtype=np.uint8))
    img.show()


def load_images(input_dir, batch_size, targets_df):
    images, batch_names, batch_targets = [], [], []
    file_names = [filename for filename in listdir(input_dir) if filename.endswith(".png")]

    for f in file_names:
        img = Image.open(join(input_dir, f))
        img = img.resize((SIDE, SIDE))
        images.append(np.array(img, dtype=K.floatx()))
        batch_names.append(f)
        batch_targets.append(targets_df.loc[f, "Target"])

        if len(batch_names) == batch_size:
            batch = np.stack(images)
            batch = preprocess_input(batch)
            if K.image_data_format() == "channels_first":
                batch = np.rollaxis(batch, 3, 1)
            yield batch, batch_names, batch_targets
            images, batch_names, batch_targets = [], [], []

    if len(batch_names) > 0:
        batch = np.stack(images)
        batch = preprocess_input(batch)
        if K.image_data_format() == "channels_first":
            batch = np.rollaxis(batch, 3, 1)
        yield batch, batch_names, batch_targets


class TerminateOnGoodAccuracy(Callback):
    """Callback that terminates training when a good accuracy is reached."""

    def __init__(self, limit_acc=0.9):
        self.limit_acc = limit_acc
        super(TerminateOnGoodAccuracy, self).__init__()

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        acc = logs.get("acc")
        if acc is not None:
            if acc >= self.limit_acc:
                print("Batch %d: accuracy %0.2f reached, terminating training" % (batch, acc))
                self.model.stop_training = True


batch_print_callback = LambdaCallback(
    on_batch_end=lambda batch, logs: print(batch, logs["loss"], logs["acc"]))


def process_batch(model, picture_layer, batch, batch_targets, seen, matched, ts, nb_epoch):
    adversarials = []
    for d, target in zip(batch, batch_targets):
        dummy = np.ones((DATA_SIZE, 1))
        target = [target - 1] * DATA_SIZE  # subtract background class
        target_cat = to_categorical(target, 1000)
        picture_layer.set_weights([d.reshape((1, -1))])
        model.fit(x=dummy, y=target_cat, epochs=nb_epoch,
                  batch_size=BATCH_SIZE,
                  callbacks=[TerminateOnGoodAccuracy(GOOD_ACC)],
                  # validation_data=[dummy, target_cat],
                  verbose=0,
                  )

        w = model.get_layer("adversarial").get_weights()[0]
        adversarials.append(d + w.reshape(SHAPE))

        seen += 1
        te = time()
        elapsed_time = (te - ts)
        average_time = elapsed_time / seen
        print("Runtime: {} sec/image. {} images seen.".format(average_time, seen))

        if seen > WARM_UP:
            if average_time > TARGET_TIME:
                nb_epoch = max(MIN_EPOCH, nb_epoch - 1)
                print("Decrease nb_epoch to", nb_epoch)
            if average_time < TARGET_TIME:
                nb_epoch = min(MAX_EPOCH, nb_epoch + 1)
                print("Increase nb_epoch to", nb_epoch)

        if VERBOSE:

            diff = (adversarials[-1] - d) / 2 * 255
            print("min {}, max {}".format(diff.min(), diff.max()))

            pred = model.predict(dummy[:1])[0]
            topk = list(pred.argsort()[-10:][::-1])
            if target[0] == topk[0]:
                matched += 1
            print("Running accuracy {:0.3f}. Target {}, Predicted {}".format(matched / seen, target[0], topk[0]))
            print([IMAGENET_LABELS[i][:15] for i in topk])

    return adversarials, seen, matched, nb_epoch


def save_images(adversarials, batch_names, output_dir):
    for adversarial, filename in zip(adversarials, batch_names):
        a = np.clip((adversarial.reshape(SHAPE) + 1.0) / 2 * 255, 0, 255)
        img = Image.fromarray(a.astype(np.uint8))
        img.save(join(output_dir, filename), format="PNG")


def rotation_layer(rotation_range, mirror=MIRROR):

    def rotation(x):
        if mirror:
            r = x[:, :, :, ::-1] if K.image_data_format() == "channels_first" else x[:, :, ::-1, :]
            x = K.switch(K.greater_equal(K.random_uniform((1,), 0, 1)[0], 0.5), lambda: r, x)

        ang = np.pi / 180 * K.random_uniform(K.shape(x)[:1], -rotation_range, rotation_range)
        x = tf.contrib.image.rotate(x, ang)
        return x

    def rotation_output_shape(input_shape):
        return input_shape
    return Lambda(rotation, output_shape=rotation_output_shape)


class MinMax(Constraint):
    """Constrains the weights to be between a lower bound and an upper bound.
    """
    def __init__(self, eps_value=0.0):
        self.eps_value = eps_value
    def __call__(self, w):
        return K.clip(w, -self.eps_value, self.eps_value)


def clip(x):
    return K.clip(x, -1, 1)


def create_model(epsilon, rotation_range):
    eps_ = epsilon / 255. * 2.
    base_model = InceptionV3(weights=None)
    base_model.load_weights(INCEPTION_PATH)
    base_model.trainable = False

    dummy_input = Input(shape=(1,))
    x0 = Dense(3 * SIDE * SIDE, use_bias=False,
               kernel_initializer="zero",
               kernel_constraint=MinMax(eps_),
               name="adversarial")(dummy_input)
    x1 = Dense(3 * SIDE * SIDE, use_bias=False,
               kernel_initializer="zero",
               trainable=False,
               name="picture")(dummy_input)

    x = Add()([x0, x1])
    x = Reshape(SHAPE)(x)
    x = rotation_layer(rotation_range)(x)
    x = Activation(clip)(x)
    model_output = base_model(x)
    model = Model(inputs=dummy_input, outputs=model_output)

    optimizer = Adam(lr=LR)
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

    picture_layer = model.get_layer("picture")

    return model, picture_layer


if __name__ == "__main__":
    ts = time()

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir",
                        required=True,
                        default="input_dir")
    parser.add_argument("--output_dir",
                        required=True,
                        default="output_dir")
    parser.add_argument("--max_epsilon",
                        required=True,
                        type=int,
                        default=10)
    args = parser.parse_args()
    INPUT_DIRECTORY = args.input_dir
    OUTPUT_DIRECTORY = args.output_dir
    MAX_EPSILON = args.max_epsilon

    with open("imagenet.json") as f:
        IMAGENET_LABELS = json.load(f)

    targets_df = load_target_class(INPUT_DIRECTORY)

    rotation_range = [15, 35, 50, 65][max(np.digitize([MAX_EPSILON], [4, 8, 12, 16])[0] - 1, 0)]
    print("Epsilon {}. Rotation range {}. Mirror {}".format(MAX_EPSILON, rotation_range, MIRROR))

    model, picture_layer = create_model(MAX_EPSILON, rotation_range)

    seen, matched = 0, 0
    nb_epoch = MAX_EPOCH
    for batch, batch_names, batch_targets in load_images(INPUT_DIRECTORY, 30, targets_df):
        adversarials, seen, matched, nb_epoch = process_batch(model, picture_layer, batch, batch_targets,
                                                              seen, matched, ts, nb_epoch)
        save_images(adversarials, batch_names, OUTPUT_DIRECTORY)

    te = time()
    elapsed_time = (te - ts)
    print("Total time: {} sec. {} images seen.".format(elapsed_time, seen))
