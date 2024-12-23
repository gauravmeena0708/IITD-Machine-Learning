import sys
sys.path.append("Sharpness-Aware-Minimization-TensorFlow")

import tensorflow as tf
tf.random.set_seed(42)
print(tf.__version__)

import tensorflow as tf

try:
    # Detect TPUs
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection
    print("Running on TPU ", tpu.master())
except ValueError:
    print("Could not connect to TPU")
    tpu = None

if tpu:
    # Connect to the TPU
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.TPUStrategy(tpu)
    print("TPU initialized")
else:
    # Default to MirroredStrategy for GPU/CPU
    strategy = tf.distribute.MirroredStrategy()
    print("Running on GPU/CPU")


print("Number of accelerators: ", strategy.num_replicas_in_sync)

import matplotlib.pyplot as plt
import time

# Load Dataset and Prepare Data Loaders
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
print(f"Training samples: {len(x_train)}")
print(f"Testing samples: {len(x_test)}")

BATCH_SIZE = 128 * strategy.num_replicas_in_sync
print(f"Batch size: {BATCH_SIZE}")
AUTO = tf.data.AUTOTUNE

def scale(image, label):
    image = tf.image.convert_image_dtype(image, tf.float32)
    label = tf.cast(label, tf.int32)
    return image, label

def augment(image, label):
    image = tf.image.resize_with_crop_or_pad(image, 40, 40)  # Add 8 pixels of padding
    image = tf.image.random_crop(image, size=[32, 32, 3])  # Random crop back to 32x32
    image = tf.image.random_brightness(image, max_delta=0.5)  # Random brightness
    image = tf.clip_by_value(image, 0., 1.)
    return image, label

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_ds = (
    train_ds
    .shuffle(1024)
    .map(scale, num_parallel_calls=AUTO)
    .map(augment, num_parallel_calls=AUTO)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)

test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_ds = (
    test_ds
    .map(scale, num_parallel_calls=AUTO)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)

# ResNet-101 Implementation for CIFAR-100
def conv_block(x, filters, kernel_size=3, stride=1, activation='relu'):
    x = tf.keras.layers.Conv2D(filters, kernel_size, strides=stride, padding='same', kernel_initializer='he_normal')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    if activation == 'relu':
        x = tf.keras.layers.ReLU()(x)
    return x

def identity_block(x, filters):
    shortcut = x  # No projection, just a skip connection
    x = conv_block(x, filters)
    x = conv_block(x, filters)
    x = tf.keras.layers.Add()([x, shortcut])
    return tf.keras.layers.ReLU()(x)

def projection_block(x, filters, stride):
    # Project the shortcut to match the shape of the main path
    shortcut = conv_block(x, filters, kernel_size=1, stride=stride, activation=None)  # Change stride to match dimensions
    x = conv_block(x, filters, stride=stride)  # Convolve and downsample
    x = conv_block(x, filters)  # Another conv block without downsampling
    x = tf.keras.layers.Add()([x, shortcut])  # Add the shortcut to the main path
    return tf.keras.layers.ReLU()(x)


def resnet101(input_shape, classes):
    inputs = tf.keras.layers.Input(shape=input_shape)
    
    x = conv_block(inputs, 64)
    
    # Stage 1
    x = projection_block(x, 64, stride=1)
    for _ in range(2):
        x = identity_block(x, 64)

    # Stage 2
    x = projection_block(x, 128, stride=2)
    for _ in range(3):
        x = identity_block(x, 128)

    # Stage 3
    x = projection_block(x, 256, stride=2)
    for _ in range(22):
        x = identity_block(x, 256)

    # Stage 4
    x = projection_block(x, 512, stride=2)
    for _ in range(2):
        x = identity_block(x, 512)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(classes, activation='softmax', kernel_initializer='he_normal')(x)

    return tf.keras.Model(inputs, outputs)

# Encapsulate SAM Logic
class SAMModel(tf.keras.Model):
    def __init__(self, resnet_model, rho=0.05):
        super(SAMModel, self).__init__()
        self.resnet_model = resnet_model
        self.rho = rho

    def train_step(self, data):
        (images, labels) = data
        e_ws = []
        with tf.GradientTape() as tape:
            predictions = self.resnet_model(images)
            loss = self.compiled_loss(labels, predictions)
        trainable_params = self.resnet_model.trainable_variables
        gradients = tape.gradient(loss, trainable_params)
        grad_norm = self._grad_norm(gradients)
        scale = self.rho / (grad_norm + 1e-12)

        for (grad, param) in zip(gradients, trainable_params):
            e_w = grad * scale
            param.assign_add(e_w)
            e_ws.append(e_w)

        with tf.GradientTape() as tape:
            predictions = self.resnet_model(images)
            loss = self.compiled_loss(labels, predictions)

        sam_gradients = tape.gradient(loss, trainable_params)
        for (param, e_w) in zip(trainable_params, e_ws):
            param.assign_sub(e_w)

        self.optimizer.apply_gradients(zip(sam_gradients, trainable_params))

        self.compiled_metrics.update_state(labels, predictions)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        (images, labels) = data
        predictions = self.resnet_model(images, training=False)
        loss = self.compiled_loss(labels, predictions)
        self.compiled_metrics.update_state(labels, predictions)
        return {m.name: m.result() for m in self.metrics}

    def _grad_norm(self, gradients):
        norm = tf.norm(
            tf.stack([
                tf.norm(grad) for grad in gradients if grad is not None
            ])
        )
        return norm

# Define Callbacks
train_callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=10,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5,
        patience=3, verbose=1
    )
]

# Initialize Model with SAM and Train It
with strategy.scope():
    model = SAMModel(resnet101(input_shape=(32, 32, 3), classes=100))
model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)
print(f"Total learnable parameters: {model.resnet_model.count_params() / 1e6} M")

start = time.time()
history = model.fit(train_ds,
                    validation_data=test_ds,
                    callbacks=train_callbacks,
                    epochs=100)
print(f"Total training time: {(time.time() - start) / 60.} minutes")

# Plot History
def plot_history(history):
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.legend()
    plt.grid()
    plt.show()

plot_history(history)

# Train a Regular ResNet-101 Model
with strategy.scope():
    model = resnet101(input_shape=(32, 32, 3), classes=100)

model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

start = time.time()
history = model.fit(train_ds,
                    validation_data=test_ds,
                    callbacks=train_callbacks,
                    epochs=200)  # 200 epochs since SAM takes two backprop steps for an update
print(f"Total training time: {(time.time() - start) / 60.} minutes")

plot_history(history)

