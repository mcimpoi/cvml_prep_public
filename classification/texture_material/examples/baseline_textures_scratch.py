import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

import tensorflow_datasets as tfds
from typing import Dict

# huggingface
import datasets
import transformers

from functools import partial
from classification.texture_material.utils import show_batch

#
# For errors like "libdevice not found at ./libdevice.10.bc"
# export XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/lib/cuda/
# /usr/lib/cuda/nvvm/libdevice  should contain libdevice.10.bc.


def get_dtd_dataset_tf(image_size: int) -> Dict[str, tf.data.Dataset]:
    train_dataset, validation_dataset = tfds.load("dtd", split=["train", "validation"])

    resize_fn = keras.layers.Resizing(image_size, image_size)

    train_dataset = (
        train_dataset.map(lambda x: (resize_fn(x["image"]) / 255.0, x["label"]))
        .shuffle(1000)
        .batch(32)
    )
    validation_dataset = (
        validation_dataset.map(lambda x: (resize_fn(x["image"]) / 255.0, x["label"]))
        .shuffle(1000)
        .batch(32)
    )

    return {"train": train_dataset, "validation": validation_dataset}


def preprocessing(examples, image_size=224):
    examples["image"] = tf.image.resize(examples["image"], (224, 224))
    return examples


def get_dtd_dataset_hf(image_size: int) -> Dict[str, tf.data.Dataset]:
    dtd_dataset = datasets.load_dataset("mcimpoi/dtd_split_1")
    data_collator = transformers.DefaultDataCollator(return_tensors="tf")

    train_ds = (
        dtd_dataset["train"]
        .map(partial(preprocessing, image_size=224))
        .to_tf_dataset(
            columns=["image", "label"],
            batch_size=32,
            shuffle=True,
            collate_fn=data_collator,
        )
    )
    val_ds = (
        dtd_dataset["validation"]
        .map(preprocessing)
        .to_tf_dataset(
            columns=["image", "label"],
            batch_size=32,
            shuffle=True,
            collate_fn=data_collator,
        )
    )
    test_ds = dtd_dataset["test"].to_tf_dataset()

    return {"train": train_ds, "validation": val_ds, "test": test_ds}


def build_uncompiled_model(
    input_img_size: int, num_output_classes: int
) -> keras.Sequential:
    model = keras.Sequential(
        [
            keras.layers.Conv2D(
                input_shape=(input_img_size, input_img_size, 3),
                filters=32,
                kernel_size=(3, 3),
                activation="relu",
            ),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Conv2D(
                filters=64,
                kernel_size=(3, 3),
                activation="relu",
            ),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Conv2D(
                filters=128,
                kernel_size=(3, 3),
                activation="relu",
            ),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Conv2D(
                filters=128,
                kernel_size=(3, 3),
                activation="relu",
            ),
            # keras.layers.Dropout(0.5),
            keras.layers.GlobalAveragePooling2D(),
            keras.layers.Dense(units=128, activation="relu"),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(units=num_output_classes, activation="softmax"),
        ]
    )

    model.summary()
    return model


def main() -> None:
    image_sz = 224
    model = build_uncompiled_model(image_sz, 47)
    model.compile(
        loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )

    dtd_data = get_dtd_dataset_hf(image_sz)

    example_batch = next(iter(dtd_data["train"]))
    show_batch(example_batch["image"], example_batch["label"], None)

    history = model.fit(
        dtd_data["train"],
        epochs=50,
        verbose=1,
        validation_data=dtd_data["validation"],
    )

    plot_accuracy(history)


def plot_accuracy(history: tf.keras.callbacks.History) -> None:
    plt.figure(figsize=(10, 8))
    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
    plt.title("Model Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Test"], loc="upper left")
    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("Model Loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Test"], loc="upper left")
    plt.show()


if __name__ == "__main__":
    main()
