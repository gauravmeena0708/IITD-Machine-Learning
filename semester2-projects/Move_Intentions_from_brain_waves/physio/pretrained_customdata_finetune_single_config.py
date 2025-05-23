#finetuner.py (Fine-tuning pre-trained model on custom data)
import os
import numpy as np
import pandas as pd
import mne
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
import config
import logging
import time

# --- Basic Setup ---
mne.set_log_level("WARNING")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
PRETRAINED_MODEL_PATH = '/home/scai/mtech/aib242286/scratch/ail721/1804/saved_models/best_model_epoch_54_val_acc_0.7091.keras'


DATA_FILES = [
    {'set': './aadhar_Handclunch_Left_15trials.set', 'csv': './aadhar_Handclunch_Left_15trials.csv', 'type': 'Left'},
    {'set': './aadhar_Handclunch_Right_15trials.set', 'csv': './aadhar_Handclunch_Right_15trials.csv', 'type': 'Right'}
]

EVENT_IDS = {'Left': 1, 'Right': 2}
CLASS_MAP = {0: 'Left', 1: 'Right'}

# --- Preprocessing Parameters ---
TARGET_SFREQ = config.SAMPLING_FREQ
logging.info(f"Using TARGET_SFREQ = {TARGET_SFREQ} Hz (from config.py)")
FMIN = config.LOW_FREQ
FMAX = config.HIGH_FREQ
TMIN = config.TMIN
TMAX = config.TMAX
BASELINE_CORRECTION = config.BASELINE_CORRECTION
MONTAGE_NAME = config.MONTAGE_NAME

# --- Fine-tuning Parameters ---
FINETUNE_SPLIT_RATIO = 0.8 # Use 80% of custom data for fine-tuning, 20% for testing
FINETUNE_EPOCHS = 100 # Number of epochs for fine-tuning (adjust as needed)
FINETUNE_BATCH_SIZE = 8 # Smaller batch size often used for fine-tuning
FINETUNE_LR = 1e-5 # Very low learning rate for fine-tuning
FINETUNE_EARLY_STOPPING_PATIENCE = 30 # Patience for early stopping during fine-tuning
FINETUNED_MODEL_SAVE_DIR = './saved_models_finetuned/' # Directory to save the fine-tuned model

# --- Helper Functions ---
def standardize_channel_name(ch_name):
    """Removes trailing dots and whitespace from channel names."""
    return ch_name.strip().rstrip('.')

def load_and_preprocess_custom_data(data_files_info, event_ids, target_sfreq, fmin, fmax, tmin, tmax, baseline, montage_name, model_expected_n_channels, model_channels_list):
    """
    Loads custom EEG data, selects channels matching the model, preprocesses, and epochs.
    (Same function as in predict_custom_data.py)
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

        # --- Select the channels the model expects ---
        try:
            current_ch_names = raw.info['ch_names']
            standardized_names_map = {name: standardize_channel_name(name) for name in current_ch_names}
            raw.rename_channels(standardized_names_map, verbose=False)
            standardized_channels_in_raw = set(raw.info['ch_names'])
            channels_to_pick = [ch for ch in model_channels_list if ch in standardized_channels_in_raw]

            if len(channels_to_pick) != model_expected_n_channels:
                 missing_model_ch = set(model_channels_list) - standardized_channels_in_raw
                 logging.error(f"File {set_path}: Not all channels required by the model ({model_expected_n_channels}) were found after standardizing names. Missing: {missing_model_ch}. Cannot proceed.")
                 return None, None
            else:
                 logging.info(f"Picking {len(channels_to_pick)} channels required by the model: {channels_to_pick}")
                 raw.pick_channels(channels_to_pick, ordered=True)

            if len(raw.info['ch_names']) != model_expected_n_channels:
                 logging.error(f"File {set_path}: Channel count is {len(raw.info['ch_names'])} after picking, expected {model_expected_n_channels}. This should not happen.")
                 return None, None
        except Exception as e:
            logging.error(f"Error during channel selection for {set_path}: {e}", exc_info=True)
            return None, None

        # --- Proceed with preprocessing ---
        original_sfreq = raw.info['sfreq']
        if original_sfreq != target_sfreq:
            logging.info(f"Resampling '{set_path}' from {original_sfreq} Hz to {target_sfreq} Hz...")
            raw.resample(target_sfreq, npad='auto')
        else:
             logging.info(f"'{set_path}' is already at target frequency {target_sfreq} Hz.")

        # --- Read events from CSV using 'start' column ---
        try:
            events_df = pd.read_csv(csv_path)
            if 'start' not in events_df.columns:
                 logging.error(f"'start' column not found in {csv_path}. Please check column names.")
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
             logging.warning("Epoching resulted in 0 epochs. Check event timing and tmin/tmax.")
             return None, None
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

# --- Main Execution ---
if __name__ == "__main__":
    # --- 1. Load Pre-trained Model Info ---
    logging.info(f"Loading pre-trained model from: {PRETRAINED_MODEL_PATH}")
    if not os.path.exists(PRETRAINED_MODEL_PATH):
        logging.error(f"Pre-trained model file not found at {PRETRAINED_MODEL_PATH}"); exit()
    try:
        # Load the model to check its input shape
        model_check = tf.keras.models.load_model(PRETRAINED_MODEL_PATH)
        input_shape = model_check.input_shape
        EXPECTED_N_CHANNELS = input_shape[1] # Should be 19
        EXPECTED_N_TIMESTEPS = input_shape[2] # Should be 401 (or whatever the model expects)
        MODEL_EXPECTED_CHANNELS = ['C3', 'C4', 'Cz', 'F3', 'F4', 'F7', 'F8', 'Fp1', 'Fp2',
                                   'Fz', 'O1', 'O2', 'P3', 'P4', 'P7', 'P8', 'Pz', 'T7', 'T8']
        if len(MODEL_EXPECTED_CHANNELS) != EXPECTED_N_CHANNELS:
             logging.error(f"Mismatch between defined MODEL_EXPECTED_CHANNELS ({len(MODEL_EXPECTED_CHANNELS)}) and model input shape ({EXPECTED_N_CHANNELS}). Update the list.")
             exit()
        logging.info(f"Pre-trained model expects input shape (Batch, Chans, Time, 1): (None, {EXPECTED_N_CHANNELS}, {EXPECTED_N_TIMESTEPS}, 1)")
        del model_check # Free up memory
    except Exception as e:
        logging.error(f"Failed to load pre-trained model or verify channels: {e}"); exit()

    # --- 2. Load and Preprocess Custom Data ---
    X_custom_all, y_custom_all = load_and_preprocess_custom_data(
        data_files_info=DATA_FILES, event_ids=EVENT_IDS,
        target_sfreq=TARGET_SFREQ, # Uses config.SAMPLING_FREQ (e.g., 160 Hz)
        fmin=FMIN, fmax=FMAX, tmin=TMIN, tmax=TMAX,
        baseline=BASELINE_CORRECTION, montage_name=MONTAGE_NAME,
        model_expected_n_channels=EXPECTED_N_CHANNELS,
        model_channels_list=MODEL_EXPECTED_CHANNELS
    )

    if X_custom_all is None or y_custom_all is None:
        logging.error("Failed to load or preprocess custom data. Exiting."); exit()

    # --- 3. Verify and Adjust Time Dimension ---
    # Check if the preprocessing result matches the model's expected timesteps
    current_n_timesteps = X_custom_all.shape[2]
    if current_n_timesteps != EXPECTED_N_TIMESTEPS:
        logging.warning(f"Adjusting time dimension: Preprocessed data has {current_n_timesteps} samples, model expects {EXPECTED_N_TIMESTEPS}.")
        logging.warning("This indicates inconsistency between current config.py parameters (TMIN/TMAX/SAMPLING_FREQ) and the parameters used to train the loaded model. Retraining the base model is recommended.")
        if current_n_timesteps > EXPECTED_N_TIMESTEPS:
            logging.info(f"Cropping time dimension from {current_n_timesteps} to {EXPECTED_N_TIMESTEPS}.")
            X_custom_all = X_custom_all[:, :, :EXPECTED_N_TIMESTEPS, :]
        else:
            logging.info(f"Padding time dimension from {current_n_timesteps} to {EXPECTED_N_TIMESTEPS} with zeros.")
            diff = EXPECTED_N_TIMESTEPS - current_n_timesteps
            pad_width = ((0, 0), (0, 0), (0, diff), (0, 0))
            X_custom_all = np.pad(X_custom_all, pad_width, mode='constant', constant_values=0)
        logging.info(f"Adjusted data shape: {X_custom_all.shape}")

    # --- 4. Split Custom Data for Fine-tuning and Testing ---
    logging.info(f"Splitting custom data ({len(y_custom_all)} samples) into {FINETUNE_SPLIT_RATIO*100:.0f}% fine-tune / {(1-FINETUNE_SPLIT_RATIO)*100:.0f}% test...")
    try:
        X_finetune, X_test_custom, y_finetune, y_test_custom = train_test_split(
            X_custom_all,
            y_custom_all,
            test_size=(1.0 - FINETUNE_SPLIT_RATIO),
            stratify=y_custom_all, # Try to keep class proportions similar
            random_state=42 # For reproducibility
        )
        logging.info(f"Fine-tune set shape: {X_finetune.shape}, Test set shape: {X_test_custom.shape}")

        # One-hot encode labels
        n_classes = len(CLASS_MAP) # Should be 2
        y_finetune_oh = tf.keras.utils.to_categorical(y_finetune, num_classes=n_classes)
        y_test_custom_oh = tf.keras.utils.to_categorical(y_test_custom, num_classes=n_classes) # For potential evaluation needs

    except Exception as e:
        logging.error(f"Error splitting or OHE data: {e}"); exit()

    # --- 5. Load Pre-trained Model for Fine-tuning ---
    logging.info("Reloading pre-trained model for fine-tuning...")
    try:
        model = tf.keras.models.load_model(PRETRAINED_MODEL_PATH)
        # Make sure all layers are trainable for fine-tuning
        model.trainable = True
        logging.info("Model layers unfrozen for fine-tuning.")

        # Compile with a low learning rate
        optimizer = tf.keras.optimizers.Adam(learning_rate=FINETUNE_LR)
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy', # Use categorical since labels are OHE
            metrics=['accuracy']
        )
        logging.info(f"Model compiled for fine-tuning with LR={FINETUNE_LR}.")
        model.summary(line_length=100)

    except Exception as e:
        logging.error(f"Error reloading/compiling model for fine-tuning: {e}"); exit()

    # --- 6. Fine-tune the Model ---
    logging.info(f"Starting fine-tuning for max {FINETUNE_EPOCHS} epochs...")

    # Define where to save the best fine-tuned model
    os.makedirs(FINETUNED_MODEL_SAVE_DIR, exist_ok=True)
    finetuned_model_filepath = os.path.join(FINETUNED_MODEL_SAVE_DIR, "finetuned_best_model_epoch_{epoch:02d}_acc_{accuracy:.4f}.keras")

    # Callbacks for fine-tuning
    finetune_callbacks = [
        EarlyStopping(
            monitor='accuracy', # Monitor training accuracy as we don't have a val split here
            patience=FINETUNE_EARLY_STOPPING_PATIENCE,
            mode='max',
            restore_best_weights=True, # Restore best weights based on training accuracy
            verbose=1
        ),
        ModelCheckpoint(
            filepath=finetuned_model_filepath,
            monitor='accuracy', # Save based on training accuracy
            save_best_only=True,
            mode='max',
            verbose=1
        )
    ]

    start_time = time.time()
    history = model.fit(
        X_finetune, y_finetune_oh,
        epochs=FINETUNE_EPOCHS,
        batch_size=FINETUNE_BATCH_SIZE,
        callbacks=finetune_callbacks,
        verbose=2 
    )
    end_time = time.time()
    logging.info(f"Fine-tuning finished in {end_time - start_time:.2f} seconds.")
    logging.info("Evaluating fine-tuned model on the custom test set...")
    try:
        X_test_custom = X_test_custom.astype(np.float32)
        loss, accuracy = model.evaluate(X_test_custom, y_test_custom_oh, verbose=0)
        logging.info(f"Evaluation on custom test set complete. Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
        predictions_prob = model.predict(X_test_custom)
        predicted_labels_int = np.argmax(predictions_prob, axis=1)

    except Exception as e:
        logging.error(f"Error during final evaluation: {e}"); exit()


    predicted_labels_str = [CLASS_MAP.get(label, f"Unknown:{label}") for label in predicted_labels_int]
    true_labels_str = [CLASS_MAP.get(label, f"Unknown:{label}") for label in y_test_custom] # Use original integer labels for comparison display

    print("\n--- Fine-tuned Model Test Results ---")
    correct_count = 0
    print(f"Test Set Size: {len(true_labels_str)}")
    for i in range(len(predicted_labels_str)):
        is_correct = predicted_labels_str[i] == true_labels_str[i]
        print(f"Test Sample {i+1}: True = {true_labels_str[i]:<5}, Predicted = {predicted_labels_str[i]:<5} {'(Correct)' if is_correct else '(Incorrect)'}")
        if is_correct: correct_count += 1

    print(f"\nOverall Accuracy on Custom Test Set (from model.evaluate): {accuracy:.4f}")
    manual_accuracy = correct_count / len(predicted_labels_str) if len(predicted_labels_str) > 0 else 0
    print(f"Overall Accuracy on Custom Test Set (manual check):       {manual_accuracy:.4f} ({correct_count}/{len(predicted_labels_str)})")

    logging.info("Fine-tuning script finished.")

