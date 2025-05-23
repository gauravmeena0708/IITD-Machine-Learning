
import os
import numpy as np
import logging
from sklearn.utils.class_weight import compute_class_weight
import config
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import tensorflow as tf

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def calculate_class_weights(y_train):
    if y_train is None or len(y_train) == 0:
        logging.warning("Empty training labels for class weight calculation.")
        return None
    unique_classes = np.unique(y_train)
    if len(unique_classes) <= 1:
        logging.info("Single class detected. Skipping class weights.")
        return None
    try:
        weights = compute_class_weight('balanced', classes=unique_classes, y=y_train)
        return dict(enumerate(weights))
    except Exception as e:
        logging.error(f"Class weight calculation failed: {e}", exc_info=True)
        return None


class EpochLoggingCallback(tf.keras.callbacks.Callback):
    def __init__(self, log_every_n=10):
        super().__init__()
        self.log_every_n = log_every_n

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.log_every_n == 0:
            logs = logs or {}
            msg = f"Epoch {epoch + 1:03d} - " + ", ".join(f"{k}: {v:.4f}" for k, v in logs.items())
            logging.info(msg)

def train_model(
    model,
    X_train, y_train_oh,
    X_val=None, y_val_oh=None,
    epochs=150, batch_size=32,
    learning_rate=1e-2,
    class_weights=None,
    early_stopping_patience=10,
    log_every_n=10,
    optimizer_name='adam',
    scheduler_name='none'
):
    if model is None:
        logging.error("No model provided to train.")
        return None, None


    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy'] 
    )


    model_save_dir = os.path.join('.', 'saved_models')
    os.makedirs(model_save_dir, exist_ok=True)
    model_filepath = os.path.join(model_save_dir, "best_model_epoch_{epoch:02d}_val_acc_{val_accuracy:.4f}.keras")



    callbacks = [
        EpochLoggingCallback(log_every_n=log_every_n),
        ModelCheckpoint(
            filepath=model_filepath,
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=False,
            mode='max',        
            verbose=1
        )
    ]


    early_stopping_monitor = 'val_accuracy'
    early_stopping_mode = 'max'

    if early_stopping_patience > 0 and X_val is not None:
        callbacks.append(
            EarlyStopping(
                monitor=early_stopping_monitor,
                patience=early_stopping_patience,
                restore_best_weights=True,
                mode=early_stopping_mode,
                verbose=1
            )
        )
    else:
         pass

    print(f"--- Starting model training for {epochs} epochs ---")
    print(f"Monitoring '{early_stopping_monitor}' for Early Stopping.")
    print(f"Saving best model based on 'val_accuracy' to: {model_save_dir}")
    history = model.fit(
        X_train, y_train_oh,
        validation_data=(X_val, y_val_oh) if X_val is not None else None,
        batch_size=batch_size,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=0
    )

    print("--- Model training finished ---")
    return model, history