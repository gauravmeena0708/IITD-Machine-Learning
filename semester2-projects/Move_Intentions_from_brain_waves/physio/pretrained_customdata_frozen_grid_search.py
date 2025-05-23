# frozen_grid_search.py
# Performs grid search for optimal hyperparameters when training only the head
# of a pre-trained model on custom data.

import os
import numpy as np
import pandas as pd
import mne
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from itertools import product # To generate hyperparameter combinations
import config
import logging
import time

# --- Basic Setup ---
mne.set_log_level("WARNING")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
PRETRAINED_MODEL_PATH = '/home/scai/mtech/aib242286/scratch/ail721/1804/saved_models/best_model_epoch_54_val_acc_0.7091.keras'
DATA_FILES = [
    {'set': './aadhar_Handclunch_Left_15trials.set', 'csv': './aadhar_Handclunch_Left_15trials.csv', 'type': 'Left'},
    {'set': './aadhar_Handclunch_Right_15trials.set', 'csv': './aadhar_Handclunch_Right_15trials.csv', 'type': 'Right'}
]

EVENT_IDS = {'Left': 1, 'Right': 2}
CLASS_MAP = {0: 'Left', 1: 'Right'}

# --- Preprocessing Parameters ---
TARGET_SFREQ = config.SAMPLING_FREQ
FMIN = config.LOW_FREQ
FMAX = config.HIGH_FREQ
TMIN = config.TMIN
TMAX = config.TMAX
BASELINE_CORRECTION = config.BASELINE_CORRECTION
MONTAGE_NAME = config.MONTAGE_NAME


SPLIT_RATIO = 0.65
LAYERS_TO_TRAIN = ['flatten', 'dense']
PARAM_GRID = {
    'HEAD_LR': [1e-2, 1e-3, 5e-3],      
    'HEAD_BATCH_SIZE': [4, 8],         
    'HEAD_EPOCHS': [50, 75, 100],     
    'HEAD_EARLY_STOPPING_PATIENCE': [20, 30] 
}


# --- Helper Functions ---
def standardize_channel_name(ch_name):
    return ch_name.strip().rstrip('.')

def load_and_preprocess_custom_data(data_files_info, event_ids, target_sfreq, fmin, fmax, tmin, tmax, baseline, montage_name, model_expected_n_channels, model_channels_list):
    all_raw = []
    all_annotations = []
    current_offset = 0.0
    for info in data_files_info:
        set_path, csv_path, event_type = info['set'], info['csv'], info['type']
        if not os.path.exists(set_path) or not os.path.exists(csv_path):
            logging.error(f"File not found: {set_path} or {csv_path}"); return None, None
        try: raw = mne.io.read_raw_eeglab(set_path, preload=True)
        except Exception as e: logging.error(f"Error loading {set_path}: {e}"); return None, None
        try:
            current_ch_names = raw.info['ch_names']
            standardized_names_map = {name: standardize_channel_name(name) for name in current_ch_names}
            raw.rename_channels(standardized_names_map, verbose=False)
            standardized_channels_in_raw = set(raw.info['ch_names'])
            channels_to_pick = [ch for ch in model_channels_list if ch in standardized_channels_in_raw]
            if len(channels_to_pick) != model_expected_n_channels:
                 missing = set(model_channels_list) - standardized_channels_in_raw
                 logging.error(f"File {set_path}: Not all required channels found. Missing: {missing}."); return None, None
            raw.pick_channels(channels_to_pick, ordered=True)
            if len(raw.info['ch_names']) != model_expected_n_channels:
                 logging.error(f"File {set_path}: Channel count mismatch after picking."); return None, None
        except Exception as e: logging.error(f"Error during channel selection for {set_path}: {e}", exc_info=True); return None, None
        original_sfreq = raw.info['sfreq']
        if original_sfreq != target_sfreq: raw.resample(target_sfreq, npad='auto')
        try:
            events_df = pd.read_csv(csv_path)
            if 'start' not in events_df.columns: logging.error(f"'start' column not found in {csv_path}."); return None, None
            onsets_sec = events_df['start'].astype(float)
            durations = np.zeros(len(onsets_sec))
            descriptions = [event_type] * len(onsets_sec)
        except Exception as e: logging.error(f"Error reading/processing {csv_path}: {e}"); return None, None
        annotations = mne.Annotations(onset=onsets_sec + current_offset, duration=durations,
                                      description=descriptions, orig_time=raw.info.get('meas_date'))
        all_raw.append(raw); all_annotations.append(annotations)
        current_offset += raw.times[-1] + (1.0 / raw.info['sfreq'])
    if not all_raw: return None, None
    raw_combined = mne.concatenate_raws(all_raw)
    combined_onset = np.concatenate([a.onset for a in all_annotations])
    combined_duration = np.concatenate([a.duration for a in all_annotations])
    combined_description = np.concatenate([a.description for a in all_annotations])
    first_meas_date = all_raw[0].info.get('meas_date')
    combined_annotations = mne.Annotations(onset=combined_onset, duration=combined_duration,
                                           description=combined_description, orig_time=first_meas_date)
    raw_combined.set_annotations(combined_annotations)
    try:
        if montage_name:
            try: montage = mne.channels.make_standard_montage(montage_name); raw_combined.set_montage(montage, on_missing='warn', verbose=False)
            except Exception as montage_e: logging.warning(f"Could not set montage '{montage_name}': {montage_e}")
        raw_combined.filter(fmin, fmax, fir_design='firwin', skip_by_annotation='edge', verbose=False)
    except Exception as e: logging.error(f"Error during filtering: {e}"); return None, None
    try:
        events, _ = mne.events_from_annotations(raw_combined, event_id=event_ids)
        if len(events) == 0: logging.error("No events found."); return None, None
        epochs = mne.Epochs(raw_combined, events, event_id=event_ids, tmin=tmin, tmax=tmax,
                            proj=False, baseline=baseline, preload=True, reject=None, verbose=False)
        if len(epochs) == 0: logging.warning("Epoching resulted in 0 epochs."); return None, None
    except Exception as e: logging.error(f"Error during epoching: {e}"); return None, None
    event_key_map = {key: i for i, key in enumerate(sorted(event_ids.keys()))}
    id_to_zero_based_map = {event_ids[key]: event_key_map[key] for key in event_ids}
    true_labels = np.array([id_to_zero_based_map[code] for code in epochs.events[:, -1]])
    X = epochs.get_data(units='uV')[..., np.newaxis]
    logging.info(f"Data loading/preprocessing complete. Prepared data shape: {X.shape}")
    return X, true_labels

# --- Main Grid Search Execution ---
if __name__ == "__main__":
    # --- 1. Load Pre-trained Model Info (Once) ---
    logging.info(f"Loading pre-trained model info: {PRETRAINED_MODEL_PATH}")
    if not os.path.exists(PRETRAINED_MODEL_PATH):
        logging.error("Pre-trained model file not found."); exit()
    try:
        model_check = tf.keras.models.load_model(PRETRAINED_MODEL_PATH)
        input_shape = model_check.input_shape
        EXPECTED_N_CHANNELS = input_shape[1]
        EXPECTED_N_TIMESTEPS = input_shape[2]
        MODEL_EXPECTED_CHANNELS = ['C3', 'C4', 'Cz', 'F3', 'F4', 'F7', 'F8', 'Fp1', 'Fp2',
                                   'Fz', 'O1', 'O2', 'P3', 'P4', 'P7', 'P8', 'Pz', 'T7', 'T8']
        if len(MODEL_EXPECTED_CHANNELS) != EXPECTED_N_CHANNELS:
             logging.error("Mismatch: MODEL_EXPECTED_CHANNELS vs model input."); exit()
        for layer_name in LAYERS_TO_TRAIN:
             try: model_check.get_layer(layer_name)
             except ValueError: logging.error(f"Layer '{layer_name}' not found in model."); exit()
        del model_check
        logging.info(f"Pre-trained model expects: {EXPECTED_N_CHANNELS} chans, {EXPECTED_N_TIMESTEPS} timesteps.")
    except Exception as e:
        logging.error(f"Failed to load pre-trained model info: {e}"); exit()

    # --- 2. Load and Preprocess Custom Data (Once) ---
    X_custom_all, y_custom_all = load_and_preprocess_custom_data(
        data_files_info=DATA_FILES, event_ids=EVENT_IDS, target_sfreq=TARGET_SFREQ,
        fmin=FMIN, fmax=FMAX, tmin=TMIN, tmax=TMAX, baseline=BASELINE_CORRECTION,
        montage_name=MONTAGE_NAME, model_expected_n_channels=EXPECTED_N_CHANNELS,
        model_channels_list=MODEL_EXPECTED_CHANNELS
    )
    if X_custom_all is None or y_custom_all is None:
        logging.error("Failed during custom data processing."); exit()

    # --- 3. Verify and Adjust Time Dimension (Once) ---
    current_n_timesteps = X_custom_all.shape[2]
    if current_n_timesteps != EXPECTED_N_TIMESTEPS:
        logging.warning(f"Adjusting time dimension: Data({current_n_timesteps}) vs Model({EXPECTED_N_TIMESTEPS}). Retraining recommended.")
        if current_n_timesteps > EXPECTED_N_TIMESTEPS:
            X_custom_all = X_custom_all[:, :, :EXPECTED_N_TIMESTEPS, :]
        else:
            diff = EXPECTED_N_TIMESTEPS - current_n_timesteps
            pad_width = ((0, 0), (0, 0), (0, diff), (0, 0))
            X_custom_all = np.pad(X_custom_all, pad_width, mode='constant', constant_values=0)

    # --- 4. Split Custom Data (Once) ---
    logging.info(f"Splitting custom data ({len(y_custom_all)} samples)...")
    try:
        X_train_custom, X_test_custom, y_train_custom, y_test_custom = train_test_split(
            X_custom_all, y_custom_all, test_size=(1.0 - SPLIT_RATIO),
            stratify=y_custom_all, random_state=42
        )
        n_classes = len(CLASS_MAP)
        y_train_custom_oh = tf.keras.utils.to_categorical(y_train_custom, num_classes=n_classes)
        y_test_custom_oh = tf.keras.utils.to_categorical(y_test_custom, num_classes=n_classes)
        logging.info(f"Train shape: {X_train_custom.shape}, Test shape: {X_test_custom.shape}")
    except Exception as e:
        logging.error(f"Error splitting or OHE data: {e}"); exit()

    # --- 5. Grid Search Loop ---
    results = []
    best_accuracy = -1.0
    best_params = None

    param_keys = PARAM_GRID.keys()
    param_values = PARAM_GRID.values()
    param_combinations = list(product(*param_values))
    total_combinations = len(param_combinations)
    logging.info(f"Starting grid search over {total_combinations} hyperparameter combinations for head training...")

    for i, combo in enumerate(param_combinations):
        current_params = dict(zip(param_keys, combo))
        lr = current_params['HEAD_LR']
        batch_size = current_params['HEAD_BATCH_SIZE']
        epochs = current_params['HEAD_EPOCHS']
        patience = current_params['HEAD_EARLY_STOPPING_PATIENCE']

        logging.info(f"\n--- Run {i+1}/{total_combinations}: Params={current_params} ---")
        tf.keras.backend.clear_session() # Clear session

        # --- Load Model and Freeze/Unfreeze Layers ---
        logging.info("Reloading pre-trained model and setting layer trainability...")
        try:
            model = tf.keras.models.load_model(PRETRAINED_MODEL_PATH)
            for layer in model.layers: layer.trainable = False # Freeze all
            for layer_name in LAYERS_TO_TRAIN: model.get_layer(layer_name).trainable = True # Unfreeze head
        except Exception as e:
            logging.error(f"Error loading/modifying model for run {i+1}: {e}"); continue

        # --- Compile ---
        try:
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                          loss='categorical_crossentropy', metrics=['accuracy'])
            logging.info(f"Model compiled with LR={lr}.")
        except Exception as e:
            logging.error(f"Error compiling model for run {i+1}: {e}"); continue

        # --- Train Head ---
        logging.info(f"Training head (Max Epochs={epochs}, Batch={batch_size})...")
        start_time = time.time()
        head_callbacks = [
            callbacks.EarlyStopping(monitor='accuracy', patience=patience, mode='max',
                                    restore_best_weights=True, verbose=0) # Monitor train acc
        ]
        history = model.fit(
            X_train_custom.astype(np.float32), y_train_custom_oh,
            epochs=epochs, batch_size=batch_size,
            callbacks=head_callbacks, verbose=0 # Suppress epoch logs
        )
        end_time = time.time()
        actual_epochs = len(history.history['accuracy'])
        logging.info(f"Head training run {i+1} finished in {end_time - start_time:.2f}s ({actual_epochs} epochs).")

        # --- Evaluate ---
        logging.info("Evaluating on custom test set...")
        try:
            loss, accuracy = model.evaluate(X_test_custom.astype(np.float32), y_test_custom_oh, verbose=0)
            logging.info(f"Run {i+1} Test Results -> Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
        except Exception as e:
            logging.error(f"Error during evaluation for run {i+1}: {e}")
            accuracy = -1 # Mark as failed

        # --- Store Results ---
        results.append({**current_params, 'test_accuracy': accuracy})

        # --- Update Best ---
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = current_params
            logging.info(f"*** New best accuracy found: {best_accuracy:.4f} with params: {best_params} ***")

    # --- 6. Print Final Results ---
    print("\n--- Grid Search Head Training Results ---")
    print("-" * 70)
    header = " | ".join(f"{key:<15}" for key in param_keys) + " | Test Accuracy"
    print(header)
    print("-" * len(header))
    # Results table
    for result in sorted(results, key=lambda x: x['test_accuracy'], reverse=True):
         row = " | ".join(f"{str(result[key]):<15}" for key in param_keys) + f" | {result['test_accuracy']:.4f}"
         print(row)
    print("-" * len(header))

    print("\n--- Best Head Training Parameters Found ---")
    if best_params:
        print(f"Parameters: {best_params}")
        print(f"Achieved Test Accuracy: {best_accuracy:.4f}")
        print("\nNote: Consider retraining the best model configuration on the full custom dataset")
        print("or using these parameters in 'frozen_single_config.py' for final model saving.")
    else:
        print("No successful runs completed.")

    logging.info("Frozen head grid search script finished.")

