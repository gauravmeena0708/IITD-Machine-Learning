import os
import numpy as np
import tensorflow as tf
import config
import preprocessor
import trainer
import evaluator
from models.qeegnet_base import build as build_qeegnet

#For single run
CONFIG = {
    'subjects': list(range(1, 110)),
    'train_ratio': 0.7,
    'val_ratio': 0.15,
    'model_name': 'qeegnet_base',
    'dropout_rate': 0.5,
    'l2_rate': 0.01,
    'f1': 8, 'd': 2, 'f2': 16, 'ke': 32, 'pooling': 8,
    'optimizer': 'adam',
    'learning_rate': 0.001,
    'epochs': 300,
    'batch_size': 32,
    'early_stopping_patience': 101,
}

def main():
    print("--- Starting EEG Classification---")
    print("\n--- Loading and Aggregating Data ---")
    try:
        X_all, y_all, label_set = preprocessor.load_and_aggregate_data(CONFIG['subjects'])
        if X_all.size == 0: raise ValueError("No data aggregated.")
        print(f"Aggregated data loaded. Shape: {X_all.shape}, Labels: {y_all.shape}, Unique: {label_set}")
        n_samples_all, n_channels, n_timesteps, _ = X_all.shape
        n_classes = len(label_set)
        input_shape = (n_channels, n_timesteps, 1)
        print(f"Inferred: n_classes={n_classes}, input_shape={input_shape}")
        target_names = [f'Class_{i}' for i in range(n_classes)]
    except Exception as e:
        print(f"Error during data aggregation: {e}")
        return

    print("\n--- Splitting Data ---")
    try:
        X_train, y_train, X_val, y_val, X_test, y_test = preprocessor.split_data(
            X_all, y_all, train_ratio=CONFIG['train_ratio'], val_ratio=CONFIG['val_ratio']
        )
        print(f"Data split. Shapes: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")
        print("\n--- One-Hot Encoding Labels ---")
        y_train_oh = tf.keras.utils.to_categorical(y_train, num_classes=n_classes)
        y_val_oh = tf.keras.utils.to_categorical(y_val, num_classes=n_classes)
        y_test_oh = tf.keras.utils.to_categorical(y_test, num_classes=n_classes)
        print(f"Label shapes after OHE: Train={y_train_oh.shape}, Val={y_val_oh.shape}, Test={y_test_oh.shape}")

    except Exception as e:
        print(f"Error during data splitting or OHE: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n--- Building Model ---")
    try:
        model = build_qeegnet(
            input_shape=input_shape,
            num_classes=n_classes,
            dropout_rate=CONFIG['dropout_rate'],
            l2_rate=CONFIG['l2_rate']
        )
        print(f"Model '{CONFIG['model_name']}' built successfully.")
        model.summary(line_length=100)
    except Exception as e:
        print(f"Error building model: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n--- Training Model (using trainer.py, WandB disabled) ---")
    try:
        trained_model, history = trainer.train_model(
            model=model,
            X_train=X_train, y_train_oh=y_train_oh,
            X_val=X_val, y_val_oh=y_val_oh,
            epochs=CONFIG['epochs'],
            batch_size=CONFIG['batch_size'],
            learning_rate=CONFIG['learning_rate'],
            early_stopping_patience=CONFIG['early_stopping_patience'],
            optimizer_name=CONFIG['optimizer'],
        )
        if trained_model is None or history is None:
             raise RuntimeError("Trainer failed to return model or history.")
        print("Model training completed via trainer.py.")

    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n--- Evaluating Model (using best weights from EarlyStopping in trainer) ---")
    try:
        test_accuracy, classification_rep, confusion_mat = evaluator.evaluate_model(
            model=trained_model,
            X_test=X_test,
            y_test_oh=y_test_oh,
            y_test=y_test,
            target_names=target_names
        )
        print("Model evaluation completed.")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        if classification_rep:
            print("Classification Report:\n", classification_rep)
        if confusion_mat is not None:
             print("Confusion Matrix:\n", confusion_mat)
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
    print("\n--- EEG Classification Script Finished ---")

if __name__ == "__main__":
    os.makedirs('./results', exist_ok=True)
    main()