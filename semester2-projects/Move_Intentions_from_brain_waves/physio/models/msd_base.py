# models/msd_base.py

import tensorflow as tf
from tensorflow.keras import layers, regularizers, Model, Input
import logging

logging.basicConfig(level=logging.INFO)

def build(input_shape, num_classes=2, dropout_rate=0.5, l2_rate=1e-4,
          filters_temporal=16, filters_separable=32, kernel_sizes=[3, 5, 7], **kwargs) -> Model:
    if len(input_shape) != 3:
        raise ValueError("Expected input shape (channels, time, 1)")

    inputs = Input(shape=input_shape)
    branches = []

    for k in kernel_sizes:
        b = layers.Conv2D(filters_temporal, (1, k), padding='same', use_bias=False,
                          kernel_regularizer=regularizers.l2(l2_rate))(inputs)
        b = layers.BatchNormalization()(b)
        b = layers.Activation('elu')(b)
        branches.append(b)

    x = layers.Concatenate(axis=-1)(branches)
    x = layers.AveragePooling2D((1, 4))(x)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.SeparableConv2D(filters_separable, (1, 16), padding='same', use_bias=False,
                               pointwise_regularizer=regularizers.l2(l2_rate),
                               depthwise_regularizer=regularizers.l2(l2_rate))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('elu')(x)
    x = layers.AveragePooling2D((1, 8))(x)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.Flatten()(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs, name="msd_base")
    return model
