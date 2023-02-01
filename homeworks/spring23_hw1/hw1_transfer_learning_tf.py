"""
CS/DS 549 Spring 2023 Programming and Model Training Assignment

The goal is to define a better model and training hyperparameters to beat the minimum
required evaluation/validation accuracy of 0.82  at the very least, but also to compete
in the class challenge for best training results.

Only edit code between the comments:
#########################
# Edit code here
# vvvvvvvvvvvvvvvvvvvvvvv
<code>
# ^^^^^^^^^^^^^^^^^^^^^^^
"""
import wandb
from wandb.keras import WandbMetricsLogger

import tensorflow as tf
from tensorflow import keras
from keras import layers
import tensorflow_datasets as tfds
from matplotlib import pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Flatten, Resizing


if __name__ == '__main__':

    # Leave entity="bu-spark-ml" and project="hw1_spring2023"
    # put your BU username in the `group=` parameter
    wandb.init(
        project="hw1_spring2023",  # Leave this as 'hw1_spring2023'
        entity="bu-spark-ml",  # Leave this
        group="kabilanm",  # <<<<<<< Put your BU username here
        notes="Transfer learning model"  # <<<<<<< You can put a short note here
    )

    """
    Use tfds to load the CIFAR10 dataset and visualize the images and train.

    The datasets used are:
    https://www.tensorflow.org/datasets/catalog/cifar10
    https://www.tensorflow.org/datasets/catalog/cifar10_corrupted

    tfds.load() whill first check if the dataset is already downloaded to the
    path in `data_dir`. If not, it will download the dataset to that path..
    """
    # Load the CIFAR10 dataset
    print("Loading CIFAR10 dataset...")
    (ds_cifar10_train, ds_cifar10_test), ds_cifar10_info = tfds.load(
        'cifar10',
        split=['train', 'test'],
        data_dir='./datasets/tensorflow_datasets',
        shuffle_files=True, # load in random order
        as_supervised=True, # Include labels
        with_info=True, # Include info
    )

    # Optionally uncomment the next 3 lines to visualize random samples from each dataset
    #fig_train = tfds.show_examples(ds_cifar10_train, ds_cifar10_info)
    #fig_test = tfds.show_examples(ds_cifar10_test, ds_cifar10_info)
    #plt.show()  # Display the plots

    def normalize_img(image, label):
        """Normalizes images: `uint8` -> `float32`."""
        return tf.cast(image, tf.float32) / 255., label

    # Prepare cifar10 training dataset
    ds_cifar10_train = ds_cifar10_train.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_cifar10_train = ds_cifar10_train.cache()     # Cache data
    ds_cifar10_train = ds_cifar10_train.shuffle(ds_cifar10_info.splits['train'].num_examples)
    ds_cifar10_train = ds_cifar10_train.batch(128)  # <<<<< To change batch size, you have to change it here
    ds_cifar10_train = ds_cifar10_train.prefetch(tf.data.AUTOTUNE)

    # Prepare cifar10 test dataset
    ds_cifar10_test = ds_cifar10_test.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_cifar10_test = ds_cifar10_test.batch(128)    # <<<<< To change batch size, you have to change it here
    ds_cifar10_test = ds_cifar10_test.cache()
    ds_cifar10_test = ds_cifar10_test.prefetch(tf.data.AUTOTUNE)

    model = Sequential()

    # Transfer learning using "DenseNet201" with weights from "ImageNet"
    inputs = keras.Input(shape=(32, 32, 3))
    base_model = keras.applications.DenseNet201(
        weights="imagenet",
        input_shape=(256, 256, 3),
        include_top=False
    )

    base_model.trainable = False # We set this to False, since we are not training the network but the last layer
    # rescale_layer = keras.layers.Rescaling(scale=1/127.5, offset=-1)

    model.add(inputs)
    model.add(Resizing(256, 256, interpolation="bilinear", crop_to_aspect_ratio=False)) # image resize to a larger resolution
    # model.add(rescale_layer)
    model.add(base_model)

    # Adding new layers to the base "DenseNet201" model to conform to our requirements
    model.add(Flatten())
    model.add(Dense(10, activation="softmax")) # final softmax layer to get predictions


    # Log the training hyper-parameters for WandB
    # If you change these in model.compile() or model.fit(), be sure to update them here.
    wandb.config = {
        #####################################
        # Edit these as desired
        # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
        # "learning_rate": 0.001,
        "optimizer": "adam",
        "epochs": 20,
        "batch_size": 128 # larger batch size to speed up the training process
        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    }

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )

    history = model.fit(
        ds_cifar10_train,
        epochs=20,
        validation_data=ds_cifar10_test,
        callbacks=[WandbMetricsLogger()]
    )

    wandb.finish()
