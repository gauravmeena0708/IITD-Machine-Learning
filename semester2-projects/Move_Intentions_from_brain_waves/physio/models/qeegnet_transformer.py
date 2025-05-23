# models/qeegnet_transformer.py

import tensorflow as tf
from tensorflow.keras import layers, Model, Input
import logging

logging.basicConfig(level=logging.INFO)

def build(input_shape, num_classes=2, dropout_rate=0.3, embed_dim=48, num_heads=4, ff_dim=96, **kwargs) -> Model:
    if len(input_shape) != 3:
        raise ValueError("Expected input shape (channels, time, 1)")

    inputs = Input(shape=input_shape)
    x = layers.Reshape((input_shape[1], input_shape[0]))(inputs)
    x = layers.Dense(embed_dim)(x)

    attn_out = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(x, x)
    x = layers.Add()([x, attn_out])
    x = layers.LayerNormalization()(x)

    ff = tf.keras.Sequential([
        layers.Dense(ff_dim, activation='relu'),
        layers.Dense(embed_dim)
    ])
    x_ff = ff(x)
    x = layers.Add()([x, x_ff])
    x = layers.LayerNormalization()(x)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs, name='qeegnet_transformer')
    return model
