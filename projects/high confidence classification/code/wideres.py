import tensorflow as tf
import numpy as np
import pandas as pd
import random
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set random seed
def set_seed(seed):
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

set_seed(42)

# Load CIFAR-100 dataset
(x_train_full, y_train_full), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()

# Normalize images
x_train_full = x_train_full.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Split training set into training and validation sets
validation_fraction = 0.1
num_validation_samples = int(validation_fraction * x_train_full.shape[0])

x_train = x_train_full[:-num_validation_samples]
y_train = y_train_full[:-num_validation_samples]

x_val = x_train_full[-num_validation_samples:]
y_val = y_train_full[-num_validation_samples:]

# Data augmentation for training data
train_datagen = ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

# No data augmentation for validation and test data
test_datagen = ImageDataGenerator()

# Define the model
def create_wide_resnet(input_shape, num_classes):
    from tensorflow.keras.applications import ResNet50

    inputs = layers.Input(shape=input_shape)
    x = layers.experimental.preprocessing.Resizing(224, 224)(inputs)
    base_model = ResNet50(include_top=False, weights=None, input_tensor=x, classes=num_classes)
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = models.Model(inputs, outputs)
    return model

model = create_wide_resnet(input_shape=(32, 32, 3), num_classes=100)

# Compile the model
optimizer = optimizers.SGD(learning_rate=0.01, momentum=0.9, decay=5e-4)
model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['sparse_categorical_accuracy']
)

# Adjust confidence threshold dynamically
def adjust_confidence_threshold(epoch):
    return min(0.95, 0.8 + epoch * 0.001)

# Custom evaluation metric
def compute_final_score(y_true, y_pred, alpha=0.99, gamma=5):
    class_correct = np.zeros(100)
    class_total = np.zeros(100)
    class_accuracies = np.zeros(100)

    for true_label, pred_label in zip(y_true, y_pred):
        if pred_label != -1:
            class_total[true_label] += 1
            if pred_label == true_label:
                class_correct[true_label] += 1

    for i in range(100):
        if class_total[i] > 0:
            class_accuracies[i] = class_correct[i] / class_total[i]
        else:
            class_accuracies[i] = 0

    high_accuracy_count = sum(
        (1 if acc >= alpha else 0) * class_total[i] for i, acc in enumerate(class_accuracies)
    )
    low_accuracy_count = sum(
        (1 if acc < alpha else 0) * class_total[i] for i, acc in enumerate(class_accuracies)
    )

    final_score = high_accuracy_count - gamma * low_accuracy_count
    return final_score

# Training loop
num_epochs = 150
batch_size = 128
patience = 10
best_val_score = -float('inf')
early_stop_counter = 0

# Prepare data generators
train_generator = train_datagen.flow(x_train, y_train, batch_size=batch_size)
val_generator = test_datagen.flow(x_val, y_val, batch_size=batch_size, shuffle=False)

# Calculate steps per epoch
steps_per_epoch = len(train_generator)
validation_steps = len(val_generator)

for epoch in range(num_epochs):
    print(f'\nEpoch {epoch+1}/{num_epochs}')
    # Adjust confidence threshold
    current_confidence_threshold = adjust_confidence_threshold(epoch)

    # Train the model
    model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=1,
        verbose=1
    )

    # Validation
    val_predictions = []
    val_true_labels = []

    for i in range(validation_steps):
        x_batch, y_batch = val_generator[i]
        predictions = model.predict(x_batch)
        confidences = np.max(predictions, axis=1)
        predicted_labels = np.argmax(predictions, axis=1)

        # Apply confidence threshold
        predicted_labels[confidences < current_confidence_threshold] = -1

        val_predictions.extend(predicted_labels)
        val_true_labels.extend(y_batch.flatten())

    val_predictions = np.array(val_predictions)
    val_true_labels = np.array(val_true_labels, dtype=int)

    # Compute validation accuracy without confidence threshold
    total_correct_predictions = np.sum(
        np.argmax(model.predict(x_val), axis=1) == y_val.flatten()
    )
    total_val_samples = y_val.shape[0]
    val_accuracy = total_correct_predictions / total_val_samples
    val_accuracy_percent = 100.0 * val_accuracy

    # Calculate standard error and confidence intervals
    SE = np.sqrt(val_accuracy * (1 - val_accuracy) / total_val_samples)
    z_scores = {
        '85%': 1.44,
        '90%': 1.645,
        '95%': 1.96,
    }
    confidence_intervals = {}
    for confidence_level, z in z_scores.items():
        margin = z * SE
        lower_bound = max(0, val_accuracy - margin)
        upper_bound = min(1, val_accuracy + margin)
        confidence_intervals[confidence_level] = (lower_bound, upper_bound)

    # Compute final score
    final_score = compute_final_score(val_true_labels, val_predictions, alpha=0.99, gamma=5)

    # Early stopping based on final_score
    if final_score > best_val_score:
        best_val_score = final_score
        early_stop_counter = 0
        model.save('best_model.h5')
        print(f'Validation score improved to {final_score}, model saved.')
    else:
        early_stop_counter += 1
        print(f'No improvement in validation score for {early_stop_counter} epochs.')

    # Print validation accuracy and confidence intervals
    print(f'Validation Accuracy: {val_accuracy_percent:.2f}%')
    for confidence_level, (lower, upper) in confidence_intervals.items():
        lower_percent = 100.0 * lower
        upper_percent = 100.0 * upper
        print(f' - {confidence_level} Confidence Interval: [{lower_percent:.2f}%, {upper_percent:.2f}%]')

    # Check for early stopping
    if early_stop_counter >= patience:
        print('Early stopping triggered.')
        break

# Load the best model
best_model = models.load_model('best_model.h5')

# Make predictions on the test set
test_generator = test_datagen.flow(x_test, batch_size=batch_size, shuffle=False)
test_steps = len(test_generator)
test_predictions = []

for i in range(test_steps):
    x_batch = test_generator[i]
    predictions = best_model.predict(x_batch)
    confidences = np.max(predictions, axis=1)
    predicted_labels = np.argmax(predictions, axis=1)

    # Apply confidence threshold
    predicted_labels[confidences < current_confidence_threshold] = -1

    test_predictions.extend(predicted_labels)

# Generate test IDs (assuming sequential IDs starting from 1)
test_ids = np.arange(1, len(test_predictions) + 1)

# Prepare the submission file
submission_df = pd.DataFrame({
    'ID': test_ids,
    'Predicted_label': test_predictions
})

# Ensure the IDs are in the correct order
submission_df.sort_values('ID', inplace=True)

# Save to CSV
submission_df.to_csv('submission.csv', index=False)

