# Following https://keras.io/examples/vision/image_classification_with_vision_transformer/
# and https://keras.io/examples/vision/vit_small_ds/

import tensorflow as tf
from tensorflow import keras
from keras import layers

from typing import Tuple, Optional


def create_vit_classifier(input_shape: Tuple[int, int, int],
                          data_augmentation: Optional[keras.Sequential],
                          patch_extractor: layers.Layer,
                          patch_encoder: layers.Layer,
                          transformer_layers: int,
                          num_heads: int,
                          projection_dim: int,
                          transformer_mlp_units: int,
                          mlp_head_units: int,
                          num_classes: int) -> keras.Model:

    inputs = layers.Input(shape=input_shape)
    augmented_inputs = data_augmentation(inputs)
    patches = patch_extractor(augmented_inputs)
    encoded_patches = patch_encoder(patches)

    for _ in range(transformer_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=transformer_mlp_units, dropout_rate=0.1)
        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])

    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)

    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)
    logits = layers.Dense(num_classes)(features)
    model = keras.Model(inputs=inputs, outputs=logits)

    return model


# Simple class patch extractor
class PatchExtractor(layers.Layer):
    def __init__(self, patch_size: int):
        super().__init__()
        self.patch_size = patch_size
    
    def call(self, images: tf.Tensor) -> tf.Tensor:
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches


# Simple class patch encoder
class PatchEncoder(layers.Layer):
    def __init__(self, num_patches: int, projection_dim: int):
        super().__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch: tf.Tensor) -> tf.Tensor:
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded


def mlp(x: tf.Tensor, hidden_units: Tuple[int, ...],
        dropout_rate: float) -> tf.Tensor:
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)

    return x
