#frozen_single_config.py
import os
import numpy as np
import pandas as pd
import mne
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
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

# --- Preprocessing Parameters
TARGET_SFREQ = config.SAMPLING_FREQ
FMIN = config.LOW_FREQ
FMAX = config.HIGH_FREQ
TMIN = config.TMIN
TMAX = config.TMAX
BASELINE_CORRECTION = config.BASELINE_CORRECTION
MONTAGE_NAME = config.MONTAGE_NAME


SPLIT_RATIO = 0.70
LAYERS_TO_TRAIN = ['flatten', 'dense'] 


HEAD_EPOCHS = 100
HEAD_BATCH_SIZE = 4
HEAD_LR = 1e-2
HEAD_EARLY_STOPPING_PATIENCE = 25
HEAD_MODEL_SAVE_DIR = './saved_models_frozen/'

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
        try:
            raw = mne.io.read_raw_eeglab(set_path, preload=True)
        except Exception as e:
            logging.error(f"Error loading {set_path}: {e}"); return None, None

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
        except Exception as e:
            logging.error(f"Error during channel selection for {set_path}: {e}", exc_info=True); return None, None

        original_sfreq = raw.info['sfreq']
        if original_sfreq != target_sfreq:
            raw.resample(target_sfreq, npad='auto')

        try:
            events_df = pd.read_csv(csv_path)
            if 'start' not in events_df.columns:
                 logging.error(f"'start' column not found in {csv_path}."); return None, None
            onsets_sec = events_df['start'].astype(float)
            durations = np.zeros(len(onsets_sec))
            descriptions = [event_type] * len(onsets_sec)
        except Exception as e:
            logging.error(f"Error reading/processing {csv_path}: {e}"); return None, None

        annotations = mne.Annotations(onset=onsets_sec + current_offset, duration=durations,
                                      description=descriptions, orig_time=raw.info.get('meas_date'))
        all_raw.append(raw)
        all_annotations.append(annotations)
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
            try:
                montage = mne.channels.make_standard_montage(montage_name)
                raw_combined.set_montage(montage, on_missing='warn', verbose=False)
            except Exception as montage_e:
                 logging.warning(f"Could not set montage '{montage_name}': {montage_e}")
        raw_combined.filter(fmin, fmax, fir_design='firwin', skip_by_annotation='edge', verbose=False)
    except Exception as e:
        logging.error(f"Error during filtering: {e}"); return None, None

    try:
        events, _ = mne.events_from_annotations(raw_combined, event_id=event_ids)
        if len(events) == 0: logging.error("No events found."); return None, None
        epochs = mne.Epochs(raw_combined, events, event_id=event_ids, tmin=tmin, tmax=tmax,
                            proj=False, baseline=baseline, preload=True, reject=None, verbose=False)
        if len(epochs) == 0: logging.warning("Epoching resulted in 0 epochs."); return None, None
    except Exception as e:
        logging.error(f"Error during epoching: {e}"); return None, None

    event_key_map = {key: i for i, key in enumerate(sorted(event_ids.keys()))}
    id_to_zero_based_map = {event_ids[key]: event_key_map[key] for key in event_ids}
    true_labels = np.array([id_to_zero_based_map[code] for code in epochs.events[:, -1]])
    X = epochs.get_data(units='uV')[..., np.newaxis]
    logging.info(f"Data loading/preprocessing complete. Prepared data shape: {X.shape}")
    return X, true_labels


# --- Main Execution ---
if __name__ == "__main__":
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
    except Exception as e:
        logging.error(f"Failed to load pre-trained model info: {e}"); exit()

    # --- 2. Load and Preprocess Custom Data ---
    X_custom_all, y_custom_all = load_and_preprocess_custom_data(
        data_files_info=DATA_FILES, event_ids=EVENT_IDS, target_sfreq=TARGET_SFREQ,
        fmin=FMIN, fmax=FMAX, tmin=TMIN, tmax=TMAX, baseline=BASELINE_CORRECTION,
        montage_name=MONTAGE_NAME, model_expected_n_channels=EXPECTED_N_CHANNELS,
        model_channels_list=MODEL_EXPECTED_CHANNELS
    )
    if X_custom_all is None or y_custom_all is None:
        logging.error("Failed during custom data processing."); exit()

    # --- 3. Verify and Adjust Time Dimension ---
    current_n_timesteps = X_custom_all.shape[2]
    if current_n_timesteps != EXPECTED_N_TIMESTEPS:
        logging.warning(f"Adjusting time dimension: Data({current_n_timesteps}) vs Model({EXPECTED_N_TIMESTEPS}). Retraining recommended.")
        if current_n_timesteps > EXPECTED_N_TIMESTEPS:
            X_custom_all = X_custom_all[:, :, :EXPECTED_N_TIMESTEPS, :]
        else:
            diff = EXPECTED_N_TIMESTEPS - current_n_timesteps
            pad_width = ((0, 0), (0, 0), (0, diff), (0, 0))
            X_custom_all = np.pad(X_custom_all, pad_width, mode='constant', constant_values=0)

    # --- 4. Split Custom Data ---
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

    # --- 5. Load Model and Freeze Base Layers ---
    logging.info("Loading pre-trained model and freezing base layers...")
    tf.keras.backend.clear_session()
    try:
        model = tf.keras.models.load_model(PRETRAINED_MODEL_PATH)

        # Freeze all layers initially
        for layer in model.layers:
            layer.trainable = False

        # Unfreeze the specified final layers
        for layer_name in LAYERS_TO_TRAIN:
            try:
                model.get_layer(layer_name).trainable = True
                logging.info(f"Layer '{layer_name}' unfrozen for training.")
            except ValueError:
                 logging.error(f"Layer '{layer_name}' specified in LAYERS_TO_TRAIN not found in model.")
                 exit()


        logging.info("Trainable status after freezing/unfreezing:")
        for layer in model.layers:
             logging.info(f"  Layer: {layer.name:<25} Trainable: {layer.trainable}")

    except Exception as e:
        logging.error(f"Error loading or modifying model layers: {e}"); exit()

    # --- 6. Compile the Model ---
    logging.info("Compiling model for head training...")
    try:
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=HEAD_LR),
                      loss='categorical_crossentropy', metrics=['accuracy'])
        model.summary(line_length=100)
    except Exception as e:
        logging.error(f"Error compiling model: {e}"); exit()


    # --- 7. Train Only the Head ---
    logging.info("Training the model head...")
    os.makedirs(HEAD_MODEL_SAVE_DIR, exist_ok=True)
    head_tuned_filepath = os.path.join(HEAD_MODEL_SAVE_DIR, "head_tuned_best.keras")
    head_callbacks = [
        callbacks.EarlyStopping(monitor='accuracy', patience=HEAD_EARLY_STOPPING_PATIENCE,
                                mode='max', restore_best_weights=True, verbose=1),
        callbacks.ModelCheckpoint(filepath=head_tuned_filepath, monitor='accuracy',
                                  save_best_only=True, mode='max', verbose=0)
    ]

    start_time = time.time()
    history = model.fit(
        X_train_custom.astype(np.float32),
        y_train_custom_oh,
        epochs=HEAD_EPOCHS,
        batch_size=HEAD_BATCH_SIZE,
        callbacks=head_callbacks,
        verbose=0
    )
    end_time = time.time()
    logging.info(f"Head training finished in {end_time - start_time:.2f} seconds. Best model saved to {head_tuned_filepath}")

    # --- 8. Evaluate the Head-Tuned Model ---
    logging.info("Evaluating head-tuned model on test set...")
    try:
        loss, accuracy = model.evaluate(X_test_custom.astype(np.float32), y_test_custom_oh, verbose=0)
        logging.info(f"Evaluation complete. Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
        predictions_prob = model.predict(X_test_custom.astype(np.float32))
        predictions_int = np.argmax(predictions_prob, axis=1)
    except Exception as e:
        logging.error(f"Error evaluating head-tuned model: {e}"); exit()

    # --- 9. Print Results ---
    print("\n--- Head-Tuned Model Test Results ---")
    print(f"Test Set Size: {len(y_test_custom)}")
    print(f"\nOverall Accuracy on Custom Test Set: {accuracy:.4f}")
    print("\nClassification Report:")
    report = classification_report(y_test_custom, predictions_int,
                                   target_names=[CLASS_MAP[i] for i in sorted(CLASS_MAP.keys())],
                                   zero_division=0)
    print(report)

    logging.info("Head fine-tuning script finished.")

