# grid_search_finetuner.py
# Performs a grid search over fine-tuning hyperparameters for a pre-trained model.

import os
import numpy as np
import pandas as pd
import mne
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from itertools import product # To generate hyperparameter combinations
import config # Import your config file for parameters
import logging
import time

# --- Basic Setup ---
mne.set_log_level("WARNING")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
# Path to your PRE-TRAINED 19-channel model
PRETRAINED_MODEL_PATH = '/home/scai/mtech/aib242286/scratch/ail721/1804/saved_models/best_model_epoch_54_val_acc_0.7091.keras'

# Paths to your custom 32-channel data files
DATA_FILES = [
    {'set': './aadhar_Handclunch_Left_15trials.set', 'csv': './aadhar_Handclunch_Left_15trials.csv', 'type': 'Left'},
    {'set': './aadhar_Handclunch_Right_15trials.set', 'csv': './aadhar_Handclunch_Right_15trials.csv', 'type': 'Right'}
]

# Event mapping
EVENT_IDS = {'Left': 1, 'Right': 2}
CLASS_MAP = {0: 'Left', 1: 'Right'}

# --- Preprocessing Parameters (Should match pre-trained model's conditions) ---
TARGET_SFREQ = config.SAMPLING_FREQ
logging.info(f"Using TARGET_SFREQ = {TARGET_SFREQ} Hz (from config.py)")
FMIN = config.LOW_FREQ
FMAX = config.HIGH_FREQ
TMIN = config.TMIN
TMAX = config.TMAX
BASELINE_CORRECTION = config.BASELINE_CORRECTION
MONTAGE_NAME = config.MONTAGE_NAME

# --- Fine-tuning Grid Search Parameters ---
FINETUNE_SPLIT_RATIO = 0.8 # Use 80% of custom data for fine-tuning, 20% for testing
PARAM_GRID = {
    'FINETUNE_LR': [1e-2, 1e-3, 1e-4, 5e-5, 1e-5], # Learning rates to try
    'FINETUNE_BATCH_SIZE': [4, 8], # Batch sizes to try
    'FINETUNE_EPOCHS': [25, 50, 75, 100], # Max epochs for each fine-tuning run
    'FINETUNE_EARLY_STOPPING_PATIENCE': [20] # Fixed patience for this example
}
# Note: Grid search can be time-consuming! Adjust grid as needed.

# --- Helper Functions (Copied from finetuner.py) ---
def standardize_channel_name(ch_name):
    """Removes trailing dots and whitespace from channel names."""
    return ch_name.strip().rstrip('.')

def load_and_preprocess_custom_data(data_files_info, event_ids, target_sfreq, fmin, fmax, tmin, tmax, baseline, montage_name, model_expected_n_channels, model_channels_list):
    """
    Loads custom EEG data, selects channels matching the model, preprocesses, and epochs.
    """
    all_raw = []
    all_annotations = []
    current_offset = 0.0

    logging.info("Loading and processing custom data files...")
    for info in data_files_info:
        set_path = info['set']
        csv_path = info['csv']
        event_type = info['type']

        if not os.path.exists(set_path) or not os.path.exists(csv_path):
            logging.error(f"File not found: {set_path} or {csv_path}")
            return None, None
        try:
            raw = mne.io.read_raw_eeglab(set_path, preload=True)
            logging.info(f"Loaded '{set_path}'. Original sfreq: {raw.info['sfreq']} Hz. Original Channels: {len(raw.ch_names)}")
        except Exception as e:
            logging.error(f"Error loading {set_path}: {e}")
            return None, None

        # Select the channels the model expects
        try:
            current_ch_names = raw.info['ch_names']
            standardized_names_map = {name: standardize_channel_name(name) for name in current_ch_names}
            raw.rename_channels(standardized_names_map, verbose=False)
            standardized_channels_in_raw = set(raw.info['ch_names'])
            channels_to_pick = [ch for ch in model_channels_list if ch in standardized_channels_in_raw]

            if len(channels_to_pick) != model_expected_n_channels:
                 missing_model_ch = set(model_channels_list) - standardized_channels_in_raw
                 logging.error(f"File {set_path}: Not all channels required by the model ({model_expected_n_channels}) were found. Missing: {missing_model_ch}.")
                 return None, None
            else:
                 logging.info(f"Picking {len(channels_to_pick)} channels required by the model.")
                 raw.pick_channels(channels_to_pick, ordered=True)

            if len(raw.info['ch_names']) != model_expected_n_channels:
                 logging.error(f"File {set_path}: Channel count is {len(raw.info['ch_names'])} after picking, expected {model_expected_n_channels}.")
                 return None, None
        except Exception as e:
            logging.error(f"Error during channel selection for {set_path}: {e}", exc_info=True)
            return None, None

        # Proceed with preprocessing
        original_sfreq = raw.info['sfreq']
        if original_sfreq != target_sfreq:
            logging.info(f"Resampling '{set_path}' from {original_sfreq} Hz to {target_sfreq} Hz...")
            raw.resample(target_sfreq, npad='auto')
        else:
             logging.info(f"'{set_path}' is already at target frequency {target_sfreq} Hz.")

        # Read events from CSV using 'start' column
        try:
            events_df = pd.read_csv(csv_path)
            if 'start' not in events_df.columns:
                 logging.error(f"'start' column not found in {csv_path}.")
                 return None, None
            onsets_sec = events_df['start'].astype(float)
            durations = np.zeros(len(onsets_sec))
            descriptions = [event_type] * len(onsets_sec)
        except Exception as e:
            logging.error(f"Error reading or processing {csv_path}: {e}")
            return None, None

        annotations = mne.Annotations(onset=onsets_sec + current_offset,
                                      duration=durations,
                                      description=descriptions,
                                      orig_time=raw.info.get('meas_date'))
        all_raw.append(raw)
        all_annotations.append(annotations)
        current_offset += raw.times[-1] + (1.0 / raw.info['sfreq'])

    if not all_raw: return None, None

    logging.info("Concatenating raw data files...")
    raw_combined = mne.concatenate_raws(all_raw)

    logging.info("Setting combined annotations...")
    combined_onset = np.concatenate([a.onset for a in all_annotations])
    combined_duration = np.concatenate([a.duration for a in all_annotations])
    combined_description = np.concatenate([a.description for a in all_annotations])
    first_meas_date = all_raw[0].info.get('meas_date')
    combined_annotations = mne.Annotations(onset=combined_onset, duration=combined_duration,
                                           description=combined_description, orig_time=first_meas_date)
    raw_combined.set_annotations(combined_annotations)

    logging.info("Applying final preprocessing (Filtering)...")
    try:
        if montage_name:
            try:
                montage = mne.channels.make_standard_montage(montage_name)
                raw_combined.set_montage(montage, on_missing='warn', verbose=False)
            except Exception as montage_e:
                 logging.warning(f"Could not set montage '{montage_name}': {montage_e}")

        raw_combined.filter(fmin, fmax, fir_design='firwin', skip_by_annotation='edge', verbose=False)
        logging.info(f"Filtered data between {fmin} Hz and {fmax} Hz.")
    except Exception as e:
        logging.error(f"Error during filtering: {e}")
        return None, None

    logging.info("Epoching data...")
    try:
        events, _ = mne.events_from_annotations(raw_combined, event_id=event_ids)
        if len(events) == 0:
            logging.error("No events found after processing annotations.")
            return None, None
        epochs = mne.Epochs(raw_combined, events, event_id=event_ids,
                            tmin=tmin, tmax=tmax, proj=False,
                            baseline=baseline, preload=True, reject=None, verbose=False)
        logging.info(f"Created {len(epochs)} epochs. Epoch time samples: {epochs.get_data().shape[2]}")
        if len(epochs) == 0:
             logging.warning("Epoching resulted in 0 epochs."); return None, None
    except Exception as e:
        logging.error(f"Error during epoching: {e}")
        return None, None

    event_key_map = {key: i for i, key in enumerate(sorted(event_ids.keys()))}
    id_to_zero_based_map = {event_ids[key]: event_key_map[key] for key in event_ids}
    true_labels = np.array([id_to_zero_based_map[code] for code in epochs.events[:, -1]])
    X = epochs.get_data(units='uV')
    X = X[..., np.newaxis]
    logging.info(f"Prepared data shape: {X.shape}")
    return X, true_labels

# --- Main Grid Search Execution ---
if __name__ == "__main__":
    # --- 1. Load Pre-trained Model Info (Once) ---
    logging.info(f"Loading pre-trained model info from: {PRETRAINED_MODEL_PATH}")
    if not os.path.exists(PRETRAINED_MODEL_PATH):
        logging.error(f"Pre-trained model file not found at {PRETRAINED_MODEL_PATH}"); exit()
    try:
        model_check = tf.keras.models.load_model(PRETRAINED_MODEL_PATH)
        input_shape = model_check.input_shape
        EXPECTED_N_CHANNELS = input_shape[1]
        EXPECTED_N_TIMESTEPS = input_shape[2]
        MODEL_EXPECTED_CHANNELS = ['C3', 'C4', 'Cz', 'F3', 'F4', 'F7', 'F8', 'Fp1', 'Fp2',
                                   'Fz', 'O1', 'O2', 'P3', 'P4', 'P7', 'P8', 'Pz', 'T7', 'T8']
        if len(MODEL_EXPECTED_CHANNELS) != EXPECTED_N_CHANNELS:
             logging.error(f"Mismatch: MODEL_EXPECTED_CHANNELS ({len(MODEL_EXPECTED_CHANNELS)}) vs model input ({EXPECTED_N_CHANNELS}).")
             exit()
        logging.info(f"Pre-trained model expects input shape: (None, {EXPECTED_N_CHANNELS}, {EXPECTED_N_TIMESTEPS}, 1)")
        del model_check # Free up memory
    except Exception as e:
        logging.error(f"Failed to load pre-trained model info: {e}"); exit()

    # --- 2. Load and Preprocess Custom Data (Once) ---
    X_custom_all, y_custom_all = load_and_preprocess_custom_data(
        data_files_info=DATA_FILES, event_ids=EVENT_IDS,
        target_sfreq=TARGET_SFREQ,
        fmin=FMIN, fmax=FMAX, tmin=TMIN, tmax=TMAX,
        baseline=BASELINE_CORRECTION, montage_name=MONTAGE_NAME,
        model_expected_n_channels=EXPECTED_N_CHANNELS,
        model_channels_list=MODEL_EXPECTED_CHANNELS
    )
    if X_custom_all is None or y_custom_all is None:
        logging.error("Failed to load or preprocess custom data. Exiting."); exit()

    # --- 3. Verify and Adjust Time Dimension (Once) ---
    current_n_timesteps = X_custom_all.shape[2]
    if current_n_timesteps != EXPECTED_N_TIMESTEPS:
        logging.warning(f"Adjusting time dimension: Data has {current_n_timesteps} samples, model expects {EXPECTED_N_TIMESTEPS}.")
        logging.warning("Inconsistency detected. Retraining the base model is recommended.")
        if current_n_timesteps > EXPECTED_N_TIMESTEPS:
            logging.info(f"Cropping time dimension to {EXPECTED_N_TIMESTEPS}.")
            X_custom_all = X_custom_all[:, :, :EXPECTED_N_TIMESTEPS, :]
        else:
            logging.info(f"Padding time dimension to {EXPECTED_N_TIMESTEPS} with zeros.")
            diff = EXPECTED_N_TIMESTEPS - current_n_timesteps
            pad_width = ((0, 0), (0, 0), (0, diff), (0, 0))
            X_custom_all = np.pad(X_custom_all, pad_width, mode='constant', constant_values=0)
        logging.info(f"Adjusted data shape: {X_custom_all.shape}")

    # --- 4. Split Custom Data (Once) ---
    logging.info(f"Splitting custom data ({len(y_custom_all)} samples)...")
    try:
        X_finetune, X_test_custom, y_finetune, y_test_custom = train_test_split(
            X_custom_all, y_custom_all,
            test_size=(1.0 - FINETUNE_SPLIT_RATIO),
            stratify=y_custom_all, random_state=42
        )
        logging.info(f"Fine-tune set shape: {X_finetune.shape}, Test set shape: {X_test_custom.shape}")
        n_classes = len(CLASS_MAP)
        y_finetune_oh = tf.keras.utils.to_categorical(y_finetune, num_classes=n_classes)
        y_test_custom_oh = tf.keras.utils.to_categorical(y_test_custom, num_classes=n_classes)
        X_test_custom = X_test_custom.astype(np.float32) # Ensure test data is float32
    except Exception as e:
        logging.error(f"Error splitting or OHE data: {e}"); exit()

    # --- 5. Grid Search Loop ---
    results = []
    best_accuracy = -1.0
    best_params = None

    # Generate all combinations of hyperparameters
    param_keys = PARAM_GRID.keys()
    param_values = PARAM_GRID.values()
    param_combinations = list(product(*param_values))
    total_combinations = len(param_combinations)
    logging.info(f"Starting grid search over {total_combinations} hyperparameter combinations...")

    for i, combo in enumerate(param_combinations):
        # Create dictionary for current hyperparameters
        current_params = dict(zip(param_keys, combo))
        lr = current_params['FINETUNE_LR']
        batch_size = current_params['FINETUNE_BATCH_SIZE']
        epochs = current_params['FINETUNE_EPOCHS']
        patience = current_params['FINETUNE_EARLY_STOPPING_PATIENCE']

        logging.info(f"\n--- Run {i+1}/{total_combinations}: Params={current_params} ---")
        tf.keras.backend.clear_session() # Clear session to avoid potential conflicts

        # --- Load Pre-trained Model (Fresh copy for each run) ---
        logging.info("Reloading pre-trained model...")
        try:
            model = tf.keras.models.load_model(PRETRAINED_MODEL_PATH)
            model.trainable = True # Unfreeze layers
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
            model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
            logging.info(f"Model reloaded and compiled with LR={lr}.")
        except Exception as e:
            logging.error(f"Error reloading/compiling model for run {i+1}: {e}")
            continue # Skip to next combination

        # --- Fine-tune ---
        logging.info(f"Starting fine-tuning (Max Epochs={epochs}, Batch={batch_size})...")
        start_time = time.time()
        # Callbacks for this run (no model saving needed here unless desired for each combo)
        finetune_callbacks = [
            EarlyStopping(monitor='accuracy', patience=patience, mode='max',
                          restore_best_weights=True, verbose=0) # Less verbose during grid search
        ]
        history = model.fit(
            X_finetune, y_finetune_oh,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=finetune_callbacks,
            verbose=0 # Suppress epoch logs during grid search
        )
        end_time = time.time()
        training_duration = end_time - start_time
        final_train_acc = history.history['accuracy'][-1]
        actual_epochs = len(history.history['accuracy'])
        logging.info(f"Fine-tuning run {i+1} finished in {training_duration:.2f}s ({actual_epochs} epochs). Final Train Acc: {final_train_acc:.4f}")

        # --- Evaluate ---
        logging.info("Evaluating on custom test set...")
        try:
            loss, accuracy = model.evaluate(X_test_custom, y_test_custom_oh, verbose=0)
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
    print("\n--- Grid Search Fine-tuning Results ---")
    print("-" * 60)
    # Header
    header = " | ".join(f"{key:<12}" for key in param_keys) + " | Test Accuracy"
    print(header)
    print("-" * len(header))
    # Results table
    for result in sorted(results, key=lambda x: x['test_accuracy'], reverse=True):
         row = " | ".join(f"{result[key]:<12}" for key in param_keys) + f" | {result['test_accuracy']:.4f}"
         print(row)
    print("-" * 60)

    print("\n--- Best Fine-tuning Parameters Found ---")
    if best_params:
        print(f"Parameters: {best_params}")
        print(f"Achieved Test Accuracy: {best_accuracy:.4f}")
    else:
        print("No successful runs completed.")

    logging.info("Grid search fine-tuning script finished.")

