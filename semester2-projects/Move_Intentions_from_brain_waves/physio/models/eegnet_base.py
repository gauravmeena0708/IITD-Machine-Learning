# models/eegnet_base.py

import tensorflow as tf
import logging
from tensorflow.keras import layers, regularizers, Model, Input

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def build(input_shape, num_classes=2, dropout_rate=0.5, l2_rate=1e-4, **kwargs) -> Model:
    """
    Builds the base EEGNet model (original architecture).

    Args:
        input_shape (tuple): Input shape of EEG data (channels, time, 1).
        num_classes (int): Number of output classes.
        dropout_rate (float): Dropout rate.
        l2_rate (float): L2 regularization factor.

    Returns:
        tf.keras.Model: Compiled EEGNet model.
    """
    if not isinstance(input_shape, tuple) or len(input_shape) != 3:
        raise ValueError(f"Expected input_shape as a tuple of (channels, time, 1), got {input_shape}")

    n_channels, n_times, _ = input_shape
    inputs = Input(shape=input_shape, name='input')

    # Block 1: Temporal Convolution + Spatial Depthwise Conv
    x = layers.Conv2D(
        filters=16,
        kernel_size=(1, 51),
        padding='same',
        use_bias=False,
        kernel_regularizer=regularizers.l2(l2_rate),
        name='conv_temporal'
    )(inputs)
    x = layers.BatchNormalization(name='bn_temporal')(x)
    x = layers.DepthwiseConv2D(
        kernel_size=(n_channels, 1),
        depth_multiplier=2,
        use_bias=False,
        depthwise_regularizer=regularizers.l2(l2_rate),
        name='conv_spatial'
    )(x)
    x = layers.BatchNormalization(name='bn_spatial')(x)
    x = layers.Activation('elu', name='act_spatial')(x)
    x = layers.AveragePooling2D(pool_size=(1, 4), name='pool1')(x)
    x = layers.Dropout(dropout_rate, name='drop1')(x)

    # Block 2: Separable Convolution
    x = layers.SeparableConv2D(
        filters=32,
        kernel_size=(1, 15),
        padding='same',
        use_bias=False,
        pointwise_regularizer=regularizers.l2(l2_rate),
        depthwise_regularizer=regularizers.l2(l2_rate),
        name='conv_separable'
    )(x)
    x = layers.BatchNormalization(name='bn_separable')(x)
    x = layers.Activation('elu', name='act_separable')(x)
    x = layers.AveragePooling2D(pool_size=(1, 8), name='pool2')(x)
    x = layers.Dropout(dropout_rate, name='drop2')(x)

    # Classification Head
    x = layers.Flatten(name='flatten')(x)
    x = layers.Dense(64, activation='elu', kernel_regularizer=regularizers.l2(l2_rate), name='dense1')(x)
    x = layers.Dropout(dropout_rate, name='drop_dense')(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='output')(x)

    model = Model(inputs, outputs, name='eegnet_base')
    logging.info(f"EEGNet Base model built with input shape {input_shape}")
    return model
