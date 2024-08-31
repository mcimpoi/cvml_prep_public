
# import datasets
import classification.texture_material.vit_classifier as vit
import tensorflow as tf
from tensorflow import keras
from keras import layers
import tensorflow_datasets as tfds

import matplotlib.pyplot as plt

PATCH_SIZE = 16
IMAGE_SIZE = 224
NUM_PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2

BATCH_SIZE: int = 16
NUM_EPOCHS: int = 10


def show_batch(image_batch, label_batch, dataset_info):
    plt.figure(figsize=(10, 10))
    for n in range(25):
        _ = plt.subplot(5, 5, n + 1)
        plt.imshow(image_batch[n].numpy().astype("uint8"))
        plt.title(dataset_info.features["labels"].names[label_batch[n]])
        plt.axis("off")


def main():
    (train_ds, val_ds, test_ds), dataset_info = tfds.load(
        "dtd", split=["train", "validation", "test"],
        with_info=True)

    x_train, y_train = train_ds.map(lambda image, label: (
        layers.Resizing(IMAGE_SIZE, IMAGE_SIZE)(image), label))

    image_batch, label_batch = next(iter(x_train.batch(25)))
    show_batch(image_batch, label_batch, dataset_info)

    data_augmentation = keras.Sequential([
        layers.Normalization(),
        layers.Resizing(IMAGE_SIZE, IMAGE_SIZE),
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(factor=0.02),
        layers.RandomZoom(
            height_factor=0.2, width_factor=0.2
        )
    ], name="data_augmentation")

    data_augmentation.layers[0].adapt(x_train)

    # model
    model = vit.create_vit_classifier(
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
        data_augmentation=data_augmentation,
        patch_extractor=vit.PatchExtractor(patch_size=PATCH_SIZE),
        patch_encoder=vit.PatchEncoder(
            num_patches=NUM_PATCHES, projection_dim=64),
        transformer_layers=2,
        num_heads=2,
        projection_dim=64,
        transformer_mlp_units=64,
        num_classes=47
    )

    model.summary()

    optimizer = keras.optimizers.Adam(learning_rate=0.001)

    model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            keras.metrics.SparseTopKCategoricalAccuracy(5, name="top-5-accuracy")]
    )

    checkpoint_filepath: str = "/tmp/checkpoint"
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        checkpoint_filepath,
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=True,
    )

    history = model.fit(
        x=x_train,
        y=y_train,
        batch_size=BATCH_SIZE,
        epochs=NUM_EPOCHS,
        validation=validation_data,
        callbacks=[checkpoint_callback],
    )


if __name__ == "__main__":
    main()