import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# Load CIFAR-100 dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()

# Normalize data to the range [0, 1]
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# One-hot encode the labels for classification
y_train = tf.keras.utils.to_categorical(y_train, 100)
y_test = tf.keras.utils.to_categorical(y_test, 100)

# Define input shape
input_shape = x_train.shape[1:]

# Encoder
def build_encoder(input_shape):
    inputs = layers.Input(shape=input_shape)
    
    # Downsample the input
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    encoded = layers.MaxPooling2D((2, 2), padding='same')(x)
    
    return models.Model(inputs, encoded, name="encoder")

# Decoder
def build_decoder(encoded_shape):
    inputs = layers.Input(shape=encoded_shape)
    
    # Upsample back to original input size
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.UpSampling2D((2, 2))(x)
    
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    
    decoded = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)  # Use sigmoid for final activation
    
    return models.Model(inputs, decoded, name="decoder")

# Classification head (Fully connected layers)
def build_classifier(encoder_output):
    x = layers.GlobalAveragePooling2D()(encoder_output)  # Global average pooling
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(100, activation='softmax')(x)  # 100 classes for CIFAR-100
    return x

# Build encoder and decoder
encoder = build_encoder(input_shape)
decoder = build_decoder(encoder.output_shape[1:])

# Autoencoder (for reconstruction)
autoencoder = models.Model(encoder.input, decoder(encoder.output), name="autoencoder")
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# Classifier (for classification)
classifier_output = build_classifier(encoder.output)
classifier = models.Model(encoder.input, classifier_output, name="classifier")
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train autoencoder (pretraining, optional)
autoencoder.fit(x_train, x_train, epochs=20, batch_size=128, validation_data=(x_test, x_test))

# Train the classifier (for classification)
history = classifier.fit(x_train, y_train, epochs=50, batch_size=128, validation_data=(x_test, y_test))
classifier.save('cifar100_classifier.h5')
# Plot training history
plt.plot(history.history['accuracy'], label='train accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.legend()
plt.show()

# Evaluate classifier on the test set
test_loss, test_acc = classifier.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc:.4f}")

