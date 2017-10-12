from keras import backend as K
from os.path import join
import numpy as np
from os import listdir
import argparse
from PIL import Image, ImageEnhance
import pandas as pd
from keras.models import load_model
from keras.applications.resnet50 import ResNet50
from keras.applications.imagenet_utils import preprocess_input as preprocess_vgg_resnet
from keras.applications.inception_v3 import preprocess_input as preprocess_inception
from datetime import datetime
np.random.seed(0)

BATCH_SIZE = 16
SIDE_INCEPTION = 299
SIDE_VGG = 224
NUM_CLASSES = 1000
INCEPTION_PATH = "./inception_v3_adv_from_tf.h5"
RESNET50_PATH = "./resnet50_weights_tf_dim_ordering_tf_kernels.h5"
VGG16_PATH = "./vgg16_weights_tf_dim_ordering_tf_kernels.h5"
N_SAMPLES = 5

VERBOSE = False
DEV_DATASET = "./dev_dataset.csv"


def load_images(input_dir, batch_size):
    images, batch_names = [], []
    filenames = [filename for filename in listdir(input_dir) if filename.endswith(".png")]
    for f in filenames:
        images.append(Image.open(join(input_dir, f)))
        batch_names.append(f)
        if len(batch_names) == batch_size:
            yield batch_names, images
            images, batch_names = [], []
    if len(batch_names) > 0:
        yield batch_names, images


def process_batch(images, models):
    def zoom(x, z):
        original_size = x.size[0]
        target_size = int(x.size[0] * z)
        x = x.resize((target_size, target_size), resample=Image.BILINEAR)
        step = abs(target_size - original_size) // 2
        if z > 1:
            return x.crop((step, step, step + original_size, step + original_size))
        elif z < 1:
            new_image = Image.fromarray(np.zeros((original_size, original_size, 3), dtype=np.uint8), mode="RGB")
            new_image.paste(x, (step, step))
            return new_image
        else:
            return x

    angles = np.random.random(N_SAMPLES) * 1
    zooms = np.random.random(N_SAMPLES) * 0.1 + 1
    contrast = np.random.uniform(0.7, 1.3, size=N_SAMPLES)
    color = np.random.uniform(0.7, 1.3, N_SAMPLES)
    sharpness = np.random.uniform(0.6, 1.0, N_SAMPLES)
    preds = []
    for angle, zm, cont, colr, shrp in zip(angles, zooms, contrast, color, sharpness):
        batch = images.copy()
        batch = [zoom(x, zm) for x in batch]
        batch = [x.rotate(angle, resample=Image.BILINEAR, expand=False) for x in batch]
        batch = [ImageEnhance.Contrast(x).enhance(cont) for x in batch]
        batch = [ImageEnhance.Color(x).enhance(colr) for x in batch]
        batch = [ImageEnhance.Sharpness(x).enhance(shrp) for x in batch]

        for model in models:
            batch_ = batch.copy()
            if model.name in {"resnet50", "vgg16"}:
                batch_ = [x.resize((SIDE_VGG, SIDE_VGG), resample=Image.BILINEAR) for x in batch_]
            batch_ = np.stack([np.array(x) for x in batch_]).astype(K.floatx())
            if model.name in {"resnet50", "vgg16"}:
                batch_ = preprocess_vgg_resnet(batch_)
            else:
                batch_ = preprocess_inception(batch_)
            if model.name=="inception_v3":
                preds.append(model.predict(batch_)[:, 1:])  # background class in adv_inception
            else:
                preds.append(model.predict(batch_))
    preds = np.stack(preds).mean(axis=0)
    labels = np.argmax(preds, axis=1) + 1  # add background class
    return labels


def save_images(batch, batch_names, output_dir):
    for im, filename in zip(batch, batch_names):
        img = Image.fromarray(im.astype(np.uint8))
        img.save(join(output_dir, filename), format="PNG")


def score(labels, true_labels=DEV_DATASET):
    df = pd.read_csv(labels, header=None, names=["ImageId", "Label"])
    df["ImageId"] = df["ImageId"].apply(lambda s: str.split(s, ".")[0])
    df = df.set_index("ImageId")

    df_true = pd.read_csv(true_labels, usecols=["ImageId", "TrueLabel", "TargetClass"], index_col="ImageId")
    df = df.join(df_true)
    return sum(df["TrueLabel"] == df["Label"]) / len(df), sum(df["TargetClass"] == df["Label"]) / len(df)


if __name__ == "__main__":
    ts = datetime.now()
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir",
                        required=True,
                        default="input_dir")
    parser.add_argument("--output_file",
                        required=True,
                        default="output_dir/output_file.csv")
    args = parser.parse_args()
    INPUT_DIRECTORY = args.input_dir
    OUTPUT_FILE = args.output_file

    models = []
    model = load_model(INCEPTION_PATH)
    models.append(model)
    model = ResNet50(weights=None, classes=NUM_CLASSES)
    model.load_weights(RESNET50_PATH)
    models.append(model)
    # model = VGG16(weights=None, classes=NUM_CLASSES)
    # model.load_weights(VGG16_PATH)
    # models.append(model)

    all_names, all_labels = [], []
    for batch_names, batch in load_images(INPUT_DIRECTORY, BATCH_SIZE):
        batch_labels = process_batch(batch, models)
        all_names.extend(batch_names)
        all_labels.extend(batch_labels)

    pd.DataFrame({0: all_names, 1: all_labels}).to_csv(OUTPUT_FILE, header=False, index=False)

    te = datetime.now()
    elapsed_time = (te - ts)
    print('Runtime: {0}'.format(elapsed_time))

    if VERBOSE:
        true_score, target_score = score(OUTPUT_FILE)
        print("True score: {:3.2f}, Target score: {:3.2f}".format(true_score, target_score))

'''
True score: 0.73, Target score: 0.00 - my tgts, InceptionV3_adv + ResNet50
True score: 0.71, Target score: 0.00 - my tgts, InceptionV3_adv
True score: 0.70, Target score: 0.01 - my tgts, InceptionV3_adv + VGG16
True score: 0.59, Target score: 0.00 - my tgts, ResNet50 + VGG16 
True score: 0.53, Target score: 0.01 - my tgts, InceptionV3 + ResNet50

True score: 0.94, Target score: 0.00 - their tgts iter, InceptionV3_adv + ResNet50
True score: 0.93, Target score: 0.00 - their tgts iter, InceptionV3_adv
True score: 0.93, Target score: 0.00 - their tgts iter, InceptionV3_adv + VGG16

True score: 0.80, Target score: 0.00 - their tgts step, InceptionV3_adv + ResNet50
True score: 0.80, Target score: 0.00 - their tgts step, InceptionV3_adv
True score: 0.81, Target score: 0.00 - their tgts step, InceptionV3_adv + VGG16
'''