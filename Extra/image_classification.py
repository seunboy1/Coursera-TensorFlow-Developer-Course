import os
import PIL
import random
import pathlib
import shutil
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D, Flatten

def download_data():
    dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
    data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
    data_dir = pathlib.Path(data_dir)

    return data_dir

def preprocessing(train_dir, val_dir = None):
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        fill_mode="nearest",
        horizontal_flip=True
    )
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(180, 180),
        batch_size=32,
        class_mode="binary"
    )

    if val_dir is not None:
        validation_datagen = ImageDataGenerator(rescale=1. / 255)

        validation_generator = validation_datagen.flow_from_directory(
            val_dir,
            target_size=(180, 180),
            batch_size=32,
            class_mode="binary"
        )

        return train_generator, validation_generator
    return train_generator

def split_data(SOURCE, TRAINING, TESTING, SPLIT_SIZE):
    x = os.listdir(SOURCE)
    files = random.sample(x, len(x))
    num = int(len(files) * SPLIT_SIZE)
    train = [files[:num], TRAINING]
    test = [files[num:], TESTING]

    for data in [train, test]:
        for img in data[0]:
            shutil.move(f"{SOURCE}/{img}", data[1])


def create_data(data_dir):
    paths = ["flowers/training", "flowers/testing"]
    for path in paths:
        for i in ["tulips", "dandelion", "roses", "sunflowers", "daisy"]:
            if os.path.exists(os.path.join(path, i)):
                pass
            else:
              os.makedirs(os.path.join(path, i))

    train_roses_dir = "flowers/training/roses"
    train_daisy_dir = "flowers/training/daisy"
    train_tulips_dir = "flowers/training/tulips"
    train_dandelion_dir = "flowers/training/dandelion"
    train_sunflowers_dir = "flowers/training/sunflowers"

    test_roses_dir = "flowers/testing/roses"
    test_daisy_dir = "flowers/testing/daisy"
    test_tulips_dir = "flowers/testing/tulips"
    test_dandelion_dir = "flowers/testing/dandelion"
    test_sunflowers_dir = "flowers/testing/sunflowers"

    split_size = .8

    split_data(os.path.join(data_dir, "roses"), train_roses_dir, test_roses_dir, split_size)
    split_data(os.path.join(data_dir, "daisy"), train_daisy_dir, test_daisy_dir, split_size)
    split_data(os.path.join(data_dir, "tulips"), train_tulips_dir, test_tulips_dir, split_size)
    split_data(os.path.join(data_dir, "dandelion"), train_dandelion_dir, test_dandelion_dir, split_size)
    split_data(os.path.join(data_dir, "sunflowers"), train_sunflowers_dir, test_sunflowers_dir, split_size)

def architecture():

    model = Sequential([
        Conv2D(32, 3, activation="relu", input_shape=(180, 180, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, 3, activation="relu"),
        MaxPooling2D((2, 2)),
        Conv2D(64, 3, activation="relu"),
        MaxPooling2D((2, 2)),
        Conv2D(128, 3, activation="relu"),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(5, activation="softmax")
    ])

    model.compile(optimizer="adam", #RMSprop(lr=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['acc'])

    return model

if __name__ == '__main__':

    data_dir = download_data()
    create_data(data_dir)
    train_dir = "flowers/training"
    val_dir = "flowers/testing"
    train_generator, validation_generator = preprocessing(train_dir, val_dir)
    model = architecture()
    model.summary()

    history = model.fit_generator(train_generator,
                                  epochs=15,
                                  verbose=1,
                                  validation_data=validation_generator)