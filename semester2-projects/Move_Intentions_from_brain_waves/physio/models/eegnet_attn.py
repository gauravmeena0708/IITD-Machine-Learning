# models/eegnet_attn.py

import tensorflow as tf
from tensorflow.keras import layers, regularizers, Model, Input
import logging

logging.basicConfig(level=logging.INFO)

def build(input_shape, num_classes=2, dropout_rate=0.5, l2_rate=1e-4, **kwargs) -> Model:
    if len(input_shape) != 3:
        raise ValueError(f"Expected input shape (channels, time, 1), got {input_shape}")

    n_channels, n_times, _ = input_shape
    inputs = Input(shape=input_shape)

    # Block 1: Temporal + Spatial Depthwise Convolution
    x = layers.Conv2D(16, (1, 51), padding='same', use_bias=False, kernel_regularizer=regularizers.l2(l2_rate))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.DepthwiseConv2D((n_channels, 1), depth_multiplier=2, use_bias=False,
                               depthwise_regularizer=regularizers.l2(l2_rate))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('elu')(x)
    x = layers.AveragePooling2D((1, 4))(x)
    x = layers.Dropout(dropout_rate)(x)

    # Block 2: Separable Conv
    x = layers.SeparableConv2D(32, (1, 15), padding='same', use_bias=False,
                               pointwise_regularizer=regularizers.l2(l2_rate),
                               depthwise_regularizer=regularizers.l2(l2_rate))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('elu')(x)
    x = layers.AveragePooling2D((1, 8))(x)
    x = layers.Dropout(dropout_rate)(x)

    # Channel Attention
    avg_pool = layers.GlobalAveragePooling2D()(x)
    dense1 = layers.Dense(32, activation='relu')(avg_pool)
    attn_weights = layers.Dense(x.shape[-1], activation='sigmoid')(dense1)
    attn_weights = layers.Reshape((1, 1, x.shape[-1]))(attn_weights)
    x = layers.Multiply()([x, attn_weights])

    # Classification Head
    x = layers.Flatten()(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs, name="eegnet_attn")
    return model
