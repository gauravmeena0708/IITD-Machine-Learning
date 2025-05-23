# eeg_models.py

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv2D, DepthwiseConv2D, SeparableConv2D,
                                     BatchNormalization, Activation, AveragePooling2D,
                                     Dropout, Flatten, Dense, Concatenate, # Added Concatenate
                                     Reshape, LayerNormalization, MultiHeadAttention, # Added MHA layers
                                     Permute) # Added Permute layer
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.regularizers import l2
import logging 

# --- Model 1: User-Provided EEGNet Variant ---
def build_eegnet(input_shape, num_classes=2, dropout_rate=0.5, l2_rate=1e-4, **kwargs):
    """User provided EEGNet variant. Accepts extra kwargs to ignore unused grid params."""
    n_channels, n_times, _ = input_shape
    logging.debug(f"Building 'build_eegnet' with Input Shape: {input_shape}, Num Classes: {num_classes}, Dropout: {dropout_rate}, L2 Rate: {l2_rate}")
    inputs = Input(shape=input_shape, name='Input')
    # Block 1
    x = Conv2D(16, (1, 51), padding='same', use_bias=False, kernel_regularizer=l2(l2_rate))(inputs)
    x = BatchNormalization()(x)
    x = DepthwiseConv2D((n_channels, 1), depth_multiplier=2, use_bias=False, depthwise_regularizer=l2(l2_rate))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = AveragePooling2D((1, 4))(x)
    x = Dropout(dropout_rate)(x)
    # Block 2
    x = SeparableConv2D(32, (1, 15), padding='same', use_bias=False, depthwise_regularizer=l2(l2_rate), pointwise_regularizer=l2(l2_rate))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = AveragePooling2D((1, 8))(x)
    x = Dropout(dropout_rate)(x)
    # Classification Head
    x = Flatten()(x)
    x = Dense(64, activation='elu', kernel_regularizer=l2(l2_rate))(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    return Model(inputs, outputs, name='build_eegnet')

# --- Model 2: EEGNet with Multi-Head Attention ---
def build_eegnet_mha(input_shape, num_classes=2, dropout_rate=0.5, l2_rate=1e-4, num_heads=4, **kwargs):
    """EEGNet variant incorporating Multi-Head Self-Attention blocks. Accepts num_heads."""
    n_channels, n_times, _ = input_shape
    logging.debug(f"Building 'build_eegnet_mha' - Input: {input_shape}, Classes: {num_classes}, Dropout: {dropout_rate}, L2: {l2_rate}, Heads: {num_heads}")
    inputs = Input(shape=input_shape, name='Input')

    # Block 1 Base (Conv -> BN -> DWConv -> BN -> Act)
    x = Conv2D(16, (1, 51), padding='same', use_bias=False, kernel_regularizer=l2(l2_rate))(inputs)
    x = BatchNormalization()(x)
    x = DepthwiseConv2D((n_channels, 1), depth_multiplier=2, use_bias=False, depthwise_regularizer=l2(l2_rate))(x)
    x = BatchNormalization()(x)
    x_act1 = Activation('elu')(x) # Output shape: (batch, Chans, Samples, 16*2=32)

    # --- Attention 1 ---
    shape_b1 = tf.keras.backend.int_shape(x_act1)
    permuted_b1 = Permute((2, 1, 3))(x_act1) # (batch, Samples, Chans, 32)
    features_dim1 = shape_b1[1] * shape_b1[3] # Chans * 32
    reshaped_attn1_input = Reshape((-1, features_dim1))(permuted_b1) # (batch, Samples, Features)
    key_dim1 = max(1, features_dim1 // num_heads)
    logging.debug(f"  MHA 1: Input={reshaped_attn1_input.shape}, Heads={num_heads}, KeyDim={key_dim1}")
    attn1_output = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim1)(query=reshaped_attn1_input, value=reshaped_attn1_input, key=reshaped_attn1_input)
    attn1_output = LayerNormalization()(attn1_output + reshaped_attn1_input) # Add & Norm
    reshaped_attn1_output = Reshape((shape_b1[2], shape_b1[1], shape_b1[3]))(attn1_output)
    permuted_attn1_output = Permute((2, 1, 3))(reshaped_attn1_output) # Back to (batch, Chans, Samples, 32)

    # Continue Block 1
    x = AveragePooling2D((1, 4))(permuted_attn1_output) # (batch, Chans, Samples/4, 32)
    x = Dropout(dropout_rate)(x)

    # Block 2 Base (SeparableConv -> BN -> Act)
    x = SeparableConv2D(32, (1, 15), padding='same', use_bias=False, depthwise_regularizer=l2(l2_rate), pointwise_regularizer=l2(l2_rate))(x)
    x = BatchNormalization()(x)
    x_act2 = Activation('elu')(x) # Output shape: (batch, Chans, Samples/4, 32)

    # --- Attention 2 ---
    shape_b2 = tf.keras.backend.int_shape(x_act2)
    permuted_b2 = Permute((2, 1, 3))(x_act2) # (batch, Samples/4, Chans, 32)
    features_dim2 = shape_b2[1] * shape_b2[3] # Chans * 32
    reshaped_attn2_input = Reshape((-1, features_dim2))(permuted_b2) # (batch, Samples/4, Features)
    key_dim2 = max(1, features_dim2 // num_heads)
    logging.debug(f"  MHA 2: Input={reshaped_attn2_input.shape}, Heads={num_heads}, KeyDim={key_dim2}")
    attn2_output = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim2)(query=reshaped_attn2_input, value=reshaped_attn2_input, key=reshaped_attn2_input)
    attn2_output = LayerNormalization()(attn2_output + reshaped_attn2_input) # Add & Norm
    reshaped_attn2_output = Reshape((shape_b2[2], shape_b2[1], shape_b2[3]))(attn2_output)
    permuted_attn2_output = Permute((2, 1, 3))(reshaped_attn2_output) # Back to (batch, Chans, Samples/4, 32)

    # Continue Block 2
    x = AveragePooling2D((1, 8))(permuted_attn2_output) # (batch, Chans, Samples/32, 32)
    x = Dropout(dropout_rate)(x)

    # Classification Head
    x = Flatten()(x)
    x = Dense(64, activation='elu', kernel_regularizer=l2(l2_rate))(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    return Model(inputs, outputs, name='build_eegnet_mha')


# --- Model 3: EEGNet-MSD (Multi-Scale Depthwise) ---
def build_eegnet_msd(input_shape, num_classes=2, dropout_rate=0.5, l2_rate=1e-4, max_norm_val=1.0, **kwargs):
    """EEGNet Multi-Scale Depthwise model. Accepts max_norm_val."""
    n_channels, n_times, _ = input_shape
    logging.debug(f"Building 'build_eegnet_msd' - Input: {input_shape}, Classes: {num_classes}, Dropout: {dropout_rate}, L2: {l2_rate}, MaxNorm: {max_norm_val}")
    inputs = Input(shape=input_shape, name='Input')

    # Default MSD parameters (can be tuned via kwargs if needed)
    kernel_sizes = kwargs.get('kernel_sizes', [16, 32, 64])
    F1 = kwargs.get('F1', 16)
    D = kwargs.get('D', 2)
    F2 = kwargs.get('F2', 64)

    multi_scale_blocks = []
    # Multi-Scale Temporal Block
    for k_len in kernel_sizes:
        branch_name = f'branch_k{k_len}'
        branch = Conv2D(F1, (1, k_len), padding='same', use_bias=False, kernel_regularizer=l2(l2_rate), name=f'{branch_name}_conv1')(inputs)
        branch = BatchNormalization(name=f'{branch_name}_bn1')(branch)
        branch = DepthwiseConv2D((n_channels, 1), depth_multiplier=D, use_bias=False, depthwise_constraint=max_norm(max_norm_val), depthwise_regularizer=l2(l2_rate), name=f'{branch_name}_dwconv')(branch)
        branch = BatchNormalization(name=f'{branch_name}_bn2')(branch)
        branch = Activation('elu', name=f'{branch_name}_act1')(branch)
        branch = AveragePooling2D((1, 4), name=f'{branch_name}_pool1')(branch)
        branch = Dropout(dropout_rate, name=f'{branch_name}_drop1')(branch)
        multi_scale_blocks.append(branch)

    merged = Concatenate(axis=-1, name='concatenate_branches')(multi_scale_blocks)

    # Separable Convolution Block
    x = SeparableConv2D(F2, (1, 16), use_bias=False, padding='same', depthwise_regularizer=l2(l2_rate), pointwise_regularizer=l2(l2_rate), name='sepconv1')(merged)
    x = BatchNormalization(name='bn3')(x)
    x = Activation('elu', name='act2')(x)
    x = AveragePooling2D((1, 8), name='pool2')(x)
    x = Dropout(dropout_rate, name='drop2')(x)

    # Additional Conv Block
    x = Conv2D(F2 * 2, (1, 16), use_bias=False, padding='same', kernel_regularizer=l2(l2_rate), name='conv2')(x)
    x = BatchNormalization(name='bn4')(x)
    x = Activation('elu', name='act3')(x)
    x = AveragePooling2D((1, 4), name='pool3')(x)
    x = Dropout(dropout_rate, name='drop3')(x)

    # Classification Head
    x = Flatten(name='flatten')(x)
    # Using max_norm_val on final Dense, as in original EEGNet-V2 example
    dense = Dense(num_classes, kernel_constraint=max_norm(max_norm_val), kernel_regularizer=l2(l2_rate), name='dense_output')(x)
    outputs = Activation('softmax', name='softmax')(dense)
    return Model(inputs, outputs, name='build_eegnet_msd')


# --- Model 4: EEGNet-MSD with Multi-Head Attention ---
def build_eegnet_msd_mha(input_shape, num_classes=2, dropout_rate=0.5, l2_rate=1e-4, max_norm_val=1.0, num_heads=4, **kwargs):
    """EEGNet-MSD model incorporating Multi-Head Self-Attention blocks."""
    n_channels, n_times, _ = input_shape
    logging.debug(f"Building 'build_eegnet_msd_mha' - Input: {input_shape}, Classes: {num_classes}, Dropout: {dropout_rate}, L2: {l2_rate}, MaxNorm: {max_norm_val}, Heads: {num_heads}")
    inputs = Input(shape=input_shape, name='Input')

    # Default MSD parameters
    kernel_sizes = kwargs.get('kernel_sizes', [16, 32, 64])
    F1 = kwargs.get('F1', 16)
    D = kwargs.get('D', 2)
    F2 = kwargs.get('F2', 64)

    multi_scale_blocks = []
    # Multi-Scale Temporal Block
    for k_len in kernel_sizes:
        branch_name = f'branch_k{k_len}'
        branch = Conv2D(F1, (1, k_len), padding='same', use_bias=False, kernel_regularizer=l2(l2_rate), name=f'{branch_name}_conv1')(inputs)
        branch = BatchNormalization(name=f'{branch_name}_bn1')(branch)
        branch = DepthwiseConv2D((n_channels, 1), depth_multiplier=D, use_bias=False, depthwise_constraint=max_norm(max_norm_val), depthwise_regularizer=l2(l2_rate), name=f'{branch_name}_dwconv')(branch)
        branch = BatchNormalization(name=f'{branch_name}_bn2')(branch)
        branch = Activation('elu', name=f'{branch_name}_act1')(branch)
        # --- No Attention within branches, pool first ---
        branch = AveragePooling2D((1, 4), name=f'{branch_name}_pool1')(branch)
        branch = Dropout(dropout_rate, name=f'{branch_name}_drop1')(branch)
        multi_scale_blocks.append(branch)

    merged = Concatenate(axis=-1, name='concatenate_branches')(multi_scale_blocks)
    # Output shape: (batch, Chans, Samples/4, F1*D*len(kernel_sizes))

    # --- Attention 1 (Applied after merging multi-scale features) ---
    shape_m1 = tf.keras.backend.int_shape(merged)
    permuted_m1 = Permute((2, 1, 3))(merged) # (batch, Samples/4, Chans, Feats)
    features_dim_m1 = shape_m1[1] * shape_m1[3] # Chans * (F1*D*len(kernel_sizes))
    reshaped_attn_m1_input = Reshape((-1, features_dim_m1))(permuted_m1) # (batch, Samples/4, Features)
    key_dim_m1 = max(1, features_dim_m1 // num_heads)
    logging.debug(f"  MSD MHA 1: Input={reshaped_attn_m1_input.shape}, Heads={num_heads}, KeyDim={key_dim_m1}")
    attn_m1_output = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim_m1)(query=reshaped_attn_m1_input, value=reshaped_attn_m1_input, key=reshaped_attn_m1_input)
    attn_m1_output = LayerNormalization()(attn_m1_output + reshaped_attn_m1_input) # Add & Norm
    reshaped_attn_m1_output = Reshape((shape_m1[2], shape_m1[1], shape_m1[3]))(attn_m1_output)
    permuted_attn_m1_output = Permute((2, 1, 3))(reshaped_attn_m1_output) # Back to (batch, Chans, Samples/4, Feats)

    # Separable Convolution Block
    x = SeparableConv2D(F2, (1, 16), use_bias=False, padding='same', depthwise_regularizer=l2(l2_rate), pointwise_regularizer=l2(l2_rate), name='sepconv1')(permuted_attn_m1_output)
    x = BatchNormalization(name='bn3')(x)
    x_act2 = Activation('elu', name='act2')(x)
    # Output shape: (batch, Chans, Samples/4, F2)

    # --- Attention 2 ---
    shape_b2 = tf.keras.backend.int_shape(x_act2)
    permuted_b2 = Permute((2, 1, 3))(x_act2)
    features_dim2 = shape_b2[1] * shape_b2[3]
    reshaped_attn2_input = Reshape((-1, features_dim2))(permuted_b2)
    key_dim2 = max(1, features_dim2 // num_heads)
    logging.debug(f"  MSD MHA 2: Input={reshaped_attn2_input.shape}, Heads={num_heads}, KeyDim={key_dim2}")
    attn2_output = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim2)(query=reshaped_attn2_input, value=reshaped_attn2_input, key=reshaped_attn2_input)
    attn2_output = LayerNormalization()(attn2_output + reshaped_attn2_input)
    reshaped_attn2_output = Reshape((shape_b2[2], shape_b2[1], shape_b2[3]))(attn2_output)
    permuted_attn2_output = Permute((2, 1, 3))(reshaped_attn2_output)

    # Continue Block 2
    x = AveragePooling2D((1, 8), name='pool2')(permuted_attn2_output)
    x = Dropout(dropout_rate, name='drop2')(x)

    # Additional Conv Block
    x = Conv2D(F2 * 2, (1, 16), use_bias=False, padding='same', kernel_regularizer=l2(l2_rate), name='conv2')(x)
    x = BatchNormalization(name='bn4')(x)
    x = Activation('elu', name='act3')(x)
    x = AveragePooling2D((1, 4), name='pool3')(x)
    x = Dropout(dropout_rate, name='drop3')(x)

    # Classification Head
    x = Flatten(name='flatten')(x)
    dense = Dense(num_classes, kernel_constraint=max_norm(max_norm_val), kernel_regularizer=l2(l2_rate), name='dense_output')(x)
    outputs = Activation('softmax', name='softmax')(dense)
    return Model(inputs, outputs, name='build_eegnet_msd_mha')


# --- Model 5: Q-EEGNet (Placeholder/Example Variant) ---
# NOTE: True quantization usually involves post-training conversion or
# quantization-aware training (QAT) using libraries like TF Lite Model Optimization.
# This is a simple architectural VARIANT, NOT actual quantization.
# We'll make it slightly lighter than build_eegnet as an example.
def build_q_eegnet(input_shape, num_classes=2, dropout_rate=0.5, l2_rate=1e-4, **kwargs):
    """Placeholder 'Quantized-like' EEGNet (Lighter variant for demonstration)."""
    n_channels, n_times, _ = input_shape
    logging.debug(f"Building 'build_q_eegnet' (Lighter Variant) - Input: {input_shape}, Classes: {num_classes}, Dropout: {dropout_rate}, L2: {l2_rate}")
    inputs = Input(shape=input_shape, name='Input')
    # Block 1 - Reduced Filters/Depth Multiplier
    F1 = 8 # Reduced from 16 in build_eegnet
    D = 1 # Reduced from 2
    kernLength = 51 # Keep same as build_eegnet
    x = Conv2D(F1, (1, kernLength), padding='same', use_bias=False, kernel_regularizer=l2(l2_rate))(inputs)
    x = BatchNormalization()(x)
    x = DepthwiseConv2D((n_channels, 1), depth_multiplier=D, use_bias=False, depthwise_regularizer=l2(l2_rate))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = AveragePooling2D((1, 4))(x)
    x = Dropout(dropout_rate)(x)
    # Block 2 - Reduced Filters
    F2 = 16 # Reduced from 32
    kernLength2 = 15 # Keep same as build_eegnet
    x = SeparableConv2D(F2, (1, kernLength2), padding='same', use_bias=False, depthwise_regularizer=l2(l2_rate), pointwise_regularizer=l2(l2_rate))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = AveragePooling2D((1, 8))(x)
    x = Dropout(dropout_rate)(x)
    # Classification Head - Reduced Dense Layer
    x = Flatten()(x)
    x = Dense(32, activation='elu', kernel_regularizer=l2(l2_rate))(x) # Reduced from 64
    x = Dropout(dropout_rate)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    return Model(inputs, outputs, name='build_q_eegnet')


# --- Model 6: Q-EEGNet with Multi-Head Attention ---
def build_q_eegnet_mha(input_shape, num_classes=2, dropout_rate=0.5, l2_rate=1e-4, num_heads=4, **kwargs):
    """Placeholder 'Quantized-like' EEGNet (Lighter Variant) with MHA blocks."""
    n_channels, n_times, _ = input_shape
    logging.debug(f"Building 'build_q_eegnet_mha' (Lighter Variant) - Input: {input_shape}, Classes: {num_classes}, Dropout: {dropout_rate}, L2: {l2_rate}, Heads: {num_heads}")
    inputs = Input(shape=input_shape, name='Input')

    # Block 1 Base (Lighter)
    F1 = 8; D = 1; kernLength = 51
    x = Conv2D(F1, (1, kernLength), padding='same', use_bias=False, kernel_regularizer=l2(l2_rate))(inputs)
    x = BatchNormalization()(x)
    x = DepthwiseConv2D((n_channels, 1), depth_multiplier=D, use_bias=False, depthwise_regularizer=l2(l2_rate))(x)
    x = BatchNormalization()(x)
    x_act1 = Activation('elu')(x) # Output shape: (batch, Chans, Samples, F1*D=8)

    # Attention 1
    shape_b1 = tf.keras.backend.int_shape(x_act1)
    permuted_b1 = Permute((2, 1, 3))(x_act1)
    features_dim1 = shape_b1[1] * shape_b1[3] # Chans * 8
    reshaped_attn1_input = Reshape((-1, features_dim1))(permuted_b1)
    key_dim1 = max(1, features_dim1 // num_heads)
    logging.debug(f"  Q MHA 1: Input={reshaped_attn1_input.shape}, Heads={num_heads}, KeyDim={key_dim1}")
    attn1_output = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim1)(query=reshaped_attn1_input, value=reshaped_attn1_input, key=reshaped_attn1_input)
    attn1_output = LayerNormalization()(attn1_output + reshaped_attn1_input)
    reshaped_attn1_output = Reshape((shape_b1[2], shape_b1[1], shape_b1[3]))(attn1_output)
    permuted_attn1_output = Permute((2, 1, 3))(reshaped_attn1_output)

    # Continue Block 1
    x = AveragePooling2D((1, 4))(permuted_attn1_output)
    x = Dropout(dropout_rate)(x)

    # Block 2 Base (Lighter)
    F2 = 16; kernLength2 = 15
    x = SeparableConv2D(F2, (1, kernLength2), padding='same', use_bias=False, depthwise_regularizer=l2(l2_rate), pointwise_regularizer=l2(l2_rate))(x)
    x = BatchNormalization()(x)
    x_act2 = Activation('elu')(x) # Output shape: (batch, Chans, Samples/4, 16)

    # Attention 2
    shape_b2 = tf.keras.backend.int_shape(x_act2)
    permuted_b2 = Permute((2, 1, 3))(x_act2)
    features_dim2 = shape_b2[1] * shape_b2[3] # Chans * 16
    reshaped_attn2_input = Reshape((-1, features_dim2))(permuted_b2)
    key_dim2 = max(1, features_dim2 // num_heads)
    logging.debug(f"  Q MHA 2: Input={reshaped_attn2_input.shape}, Heads={num_heads}, KeyDim={key_dim2}")
    attn2_output = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim2)(query=reshaped_attn2_input, value=reshaped_attn2_input, key=reshaped_attn2_input)
    attn2_output = LayerNormalization()(attn2_output + reshaped_attn2_input)
    reshaped_attn2_output = Reshape((shape_b2[2], shape_b2[1], shape_b2[3]))(attn2_output)
    permuted_attn2_output = Permute((2, 1, 3))(reshaped_attn2_output)

    # Continue Block 2
    x = AveragePooling2D((1, 8))(permuted_attn2_output)
    x = Dropout(dropout_rate)(x)

    # Classification Head (Lighter)
    x = Flatten()(x)
    x = Dense(32, activation='elu', kernel_regularizer=l2(l2_rate))(x) # Reduced dense
    x = Dropout(dropout_rate)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    return Model(inputs, outputs, name='build_q_eegnet_mha')