# models/qeegnet_base.py

import tensorflow as tf
from tensorflow.keras import layers, regularizers, Model, Input
import logging

logging.basicConfig(level=logging.INFO)

def build(input_shape, num_classes=2, dropout_rate=0.3, l2_rate=1e-4, **kwargs) -> Model:
    """
    Builds the base Q-EEGNet model (simplified EEGNet with fewer filters and ReLU activations).

    Args:
        input_shape (tuple): Shape of input data (channels, time, 1).
        num_classes (int): Number of output classes.
        dropout_rate (float): Dropout rate for regularization.
        l2_rate (float): L2 regularization factor.

    Returns:
        tf.keras.Model: The compiled Q-EEGNet model.
    """
    if len(input_shape) != 3:
        raise ValueError("Expected input shape (channels, time, 1)")

    n_channels, n_times, _ = input_shape
    inputs = Input(shape=input_shape)

    # Block 1: Temporal + Depthwise Spatial Conv
    x = layers.Conv2D(8, (1, 64), padding='same', use_bias=False,
                      kernel_regularizer=regularizers.l2(l2_rate))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.DepthwiseConv2D((n_channels, 1), depth_multiplier=1, use_bias=False,
                               depthwise_regularizer=regularizers.l2(l2_rate))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.AveragePooling2D((1, 4))(x)
    x = layers.Dropout(dropout_rate)(x)

    # Block 2: Separable Conv
    x = layers.SeparableConv2D(16, (1, 16), padding='same', use_bias=False,
                               pointwise_regularizer=regularizers.l2(l2_rate),
                               depthwise_regularizer=regularizers.l2(l2_rate))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.AveragePooling2D((1, 8))(x)
    x = layers.Dropout(dropout_rate)(x)

    # Classification
    x = layers.Flatten()(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs, name='qeegnet_base')
    return model
