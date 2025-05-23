#preprocessor.py
import logging
import numpy as np
import tensorflow as tf
import mne
from sklearn.model_selection import train_test_split
import data_loader
import config      

mne.set_log_level("WARNING")


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
def standardize_channel_name(ch_name):
    #handle . in channel names
    return ch_name.strip().rstrip('.')


def preprocess_raw_to_epochs(raw, event_id, tmin, tmax, low_freq, high_freq, baseline_correction, montage_name):
    #Preprocesses with selected channels into Epochs.
    try:
        if montage_name:
            try:
                montage = mne.channels.make_standard_montage(montage_name)
                raw.set_montage(montage, on_missing='warn', verbose=False)
            except Exception as montage_e:
                 logging.warning(f"Could not set montage '{montage_name}': {montage_e}")

        logging.debug(f"Filtering {len(raw.ch_names)} channels between {low_freq}-{high_freq} Hz.")
        raw.filter(low_freq, high_freq, fir_design='firwin', skip_by_annotation='edge', verbose=False)

        if config.DATASET_SOURCE == "EEGBCI":
            events, _ = mne.events_from_annotations(raw, event_id=event_id, verbose=False)
            logging.debug(f"Found {len(events)} events for EEGBCI.")
        else:
             logging.error(f"Unsupported DATASET_SOURCE '{config.DATASET_SOURCE}' for event extraction.")
             return None

        if len(events) == 0:
            logging.warning("No relevant events found in Raw object after filtering annotations.")
            return None


        epochs = mne.Epochs(
            raw, events, event_id=event_id,
            tmin=tmin, tmax=tmax,
            baseline=baseline_correction,
            preload=True,
            reject=None,
            reject_by_annotation=True, 
            picks='eeg', 
            verbose=False
        )

        if len(epochs) == 0:
             logging.warning("All epochs were dropped, possibly due to annotations or event timing. Check data quality and tmin/tmax.")
             return None

        logging.debug(f"Created {len(epochs)} epochs with tmin={tmin}, tmax={tmax}.")
        return epochs

    except Exception as e:
        logging.error(f"Error in preprocess_raw_to_epochs: {e}", exc_info=True)
        return None

def prepare_data_for_model(epochs, event_id):
    #Extracts data (X) and labels (y) from MNE Epochs object, ready for model input.
    if epochs is None:
        return None, None, 0
    try:
        event_key_to_index = {k: i for i, k in enumerate(sorted(event_id))}
        label_map = {epochs.event_id[k]: event_key_to_index[k] for k in event_key_to_index if k in epochs.event_id}

        valid_event_indices = [i for i, code in enumerate(epochs.events[:, -1]) if code in label_map]
        if not valid_event_indices:
            logging.warning("No epochs found with valid event IDs matching the event_id dictionary.")
            return None, None, 0

        labels = np.array([label_map[code] for i, code in enumerate(epochs.events[:, -1]) if i in valid_event_indices])
        num_classes = len(np.unique(labels))

        X = epochs.get_data(picks='eeg', units='uV')[valid_event_indices]
        X = X[..., np.newaxis]

        logging.debug(f"Prepared data X shape: {X.shape}, y shape: {labels.shape}, num_classes: {num_classes}")
        return X, labels, num_classes

    except Exception as e:
        logging.error(f"Error preparing data for model: {e}", exc_info=True)
        return None, None, 0

def load_and_aggregate_data(subjects):
    all_X = []
    all_y = []
    label_set = set()
    target_n_times = None

    # Common Channels
    CHANNELS_TO_KEEP = ['C3', 'C4', 'Cz', 'F3', 'F4', 'F7', 'F8', 'Fp1', 'Fp2',
                        'Fz', 'O1', 'O2', 'P3', 'P4', 'P7', 'P8', 'Pz', 'T7', 'T8']

    N_CHANNELS_EXPECTED = len(CHANNELS_TO_KEEP)
    logging.info(f"Starting data aggregation for {len(subjects)} subjects using {N_CHANNELS_EXPECTED} common channels.")

    for sid in subjects:
        logging.info(f"Processing Subject {sid}...")
        raw = data_loader.load_raw_data(sid, config.RUNS)
        if raw is None:
            logging.warning(f"Subject {sid}: Failed to load raw data. Skipping.")
            continue
        logging.debug(f"Subject {sid}: Loaded raw data with {len(raw.ch_names)} original channels.")
        try:
            current_ch_names = raw.info['ch_names']
            standardized_names_map = {name: standardize_channel_name(name) for name in current_ch_names}
            raw.rename_channels(standardized_names_map, verbose=False)
            logging.debug(f"Subject {sid}: Standardized channel names.")

            standardized_channels_in_raw = set(raw.info['ch_names']) 
            channels_to_pick = [ch for ch in CHANNELS_TO_KEEP if ch in standardized_channels_in_raw]

            if len(channels_to_pick) != N_CHANNELS_EXPECTED:
                 missing_common = set(CHANNELS_TO_KEEP) - standardized_channels_in_raw
                 logging.error(f"Subject {sid}: Not all {N_CHANNELS_EXPECTED} required common channels found after standardizing names. Missing: {missing_common}. Skipping.")
                 continue
            else:
                 logging.debug(f"Subject {sid}: Picking {len(channels_to_pick)} common channels: {channels_to_pick}")
                 raw.pick_channels(channels_to_pick, ordered=True)

            if len(raw.info['ch_names']) != N_CHANNELS_EXPECTED:
                 logging.error(f"Subject {sid}: Channel count is {len(raw.info['ch_names'])} after picking, expected {N_CHANNELS_EXPECTED}. This should not happen. Skipping.")
                 continue

        except Exception as e:
            logging.error(f"Subject {sid}: Error during channel standardization/selection: {e}", exc_info=True)
            continue

        logging.debug(f"Subject {sid}: Preprocessing raw data to epochs...")
        epochs = preprocess_raw_to_epochs(
            raw,
            config.EVENT_ID,
            config.TMIN,
            config.TMAX,
            config.LOW_FREQ,
            config.HIGH_FREQ,
            config.BASELINE_CORRECTION,
            config.MONTAGE_NAME
        )

        if epochs is None:
            logging.warning(f"Subject {sid}: No epochs extracted after preprocessing. Skipping.")
            continue
        logging.debug(f"Subject {sid}: Extracted {len(epochs)} epochs.")

        logging.debug(f"Subject {sid}: Preparing epoch data for model...")
        X, y, _ = prepare_data_for_model(epochs, config.EVENT_ID)
        if X is None or y is None:
            logging.warning(f"Subject {sid}: Failed to prepare data for model from epochs. Skipping.")
            continue
        logging.debug(f"Subject {sid}: Prepared data X shape: {X.shape}, y shape: {y.shape}")

        if target_n_times is None:
            target_n_times = X.shape[2]
            logging.info(f"Setting target number of time steps to {target_n_times} based on Subject {sid}.")
        elif X.shape[2] != target_n_times:
            logging.warning(f"Subject {sid}: Time dimension mismatch ({X.shape[2]} vs target {target_n_times}). Adjusting...")
            diff = target_n_times - X.shape[2]
            if diff > 0:
                pad_width = ((0, 0), (0, 0), (0, diff), (0, 0))
                X = np.pad(X, pad_width, mode='constant')
                logging.debug(f"Subject {sid}: Padded time dimension. New X shape: {X.shape}")
            else:
                X = X[:, :, :target_n_times, :]
                logging.debug(f"Subject {sid}: Cropped time dimension. New X shape: {X.shape}")

        all_X.append(X)
        all_y.append(y)
        label_set.update(np.unique(y))

    if not all_X:
        logging.error("No data successfully processed and aggregated for any subject. Check channel names, data paths, and preprocessing steps.")
        n_timesteps_fallback = int((config.TMAX - config.TMIN) * config.SAMPLING_FREQ) if target_n_times is None else target_n_times
        return np.empty((0, N_CHANNELS_EXPECTED, n_timesteps_fallback, 1)), np.empty((0,)), set()

    X_aggregated = np.concatenate(all_X)
    y_aggregated = np.concatenate(all_y)
    logging.info(f"Finished aggregation. Final X shape: {X_aggregated.shape}, y shape: {y_aggregated.shape}")
    logging.info(f"Unique labels in aggregated data: {label_set}")

    if X_aggregated.shape[1] != N_CHANNELS_EXPECTED:
         logging.error(f"Aggregated data has incorrect channel dimension ({X_aggregated.shape[1]}), expected {N_CHANNELS_EXPECTED}.")
         return None, None, None


    return X_aggregated, y_aggregated, label_set


def augment_data(X, y, noise_factor, num_copies, seed):
    if num_copies <= 0:
        return X, y
    np.random.seed(seed)
    augmented_X, augmented_y = [X], [y]
    for _ in range(num_copies):
        noise = np.random.normal(loc=0.0, scale=1.0, size=X.shape)
        std = np.std(X, axis=(1, 2), keepdims=True); 
        std[std == 0] = 1.0
        X_noisy = X + noise_factor * std * noise
        augmented_X.append(X_noisy); 
        augmented_y.append(y)
    X_all, y_all = np.concatenate(augmented_X), np.concatenate(augmented_y)
    idx = np.random.permutation(len(X_all))
    return X_all[idx], y_all[idx]

def one_hot_encode_labels(num_classes, *label_arrays):
    results = []
    for labels in label_arrays:
        if labels is not None: results.append(tf.keras.utils.to_categorical(labels, num_classes))
        else: results.append(None)
    return results

def split_subjects(subjects, train_pct, val_pct, test_pct, seed):
    np.random.seed(seed)
    np.random.shuffle(subjects)
    n = len(subjects)
    n_train, n_val = int(n * train_pct), int(n * val_pct)
    return (list(subjects[:n_train]), list(subjects[n_train:n_train + n_val]), list(subjects[n_train + n_val:]))

def log_sample_distribution(y, label=""):
    if y is None: print(f"{label} set is None"); return
    unique, counts = np.unique(y, return_counts=True)
    print(f"\n {label} Set Distribution:"); print(f"Total samples: {len(y)}")
    for cls, count in zip(unique, counts): print(f"  Class {cls}: {count}")

def log_per_subject_distribution(subjects, label=""):
    print(f"\n{label} Set Per-Subject Class Distribution (Based on reloading raw data):")
    CHANNELS_TO_LOG = ['C3', 'C4', 'Cz', 'F3', 'F4', 'F7', 'F8', 'Fp1', 'Fp2',
                       'Fz', 'O1', 'O2', 'P3', 'P4', 'P7', 'P8', 'Pz', 'T7', 'T8']
    for sid in subjects:
        raw = data_loader.load_raw_data(sid, config.RUNS)
        if raw is None: print(f"  Subject {sid}: Failed to load"); continue
        try:
            raw.rename_channels({name: standardize_channel_name(name) for name in raw.info['ch_names']}, verbose=False)
            raw.pick_channels(CHANNELS_TO_LOG, ordered=True)
        except Exception as log_pick_e:
            print(f"  Subject {sid}: Failed to pick channels for logging: {log_pick_e}"); 
            continue

        epochs = preprocess_raw_to_epochs(raw, config.EVENT_ID, config.TMIN, config.TMAX,
                                          config.LOW_FREQ, config.HIGH_FREQ, config.BASELINE_CORRECTION,
                                          config.MONTAGE_NAME)
        if epochs is None: 
            print(f"  Subject {sid}: Failed to preprocess"); 
            continue
        _, y, _ = prepare_data_for_model(epochs, config.EVENT_ID)
        if y is None: 
            print(f"  Subject {sid}: No valid labels"); 
            continue
        unique, counts = np.unique(y, return_counts=True)
        dist = ", ".join([f"Class {cls}: {cnt}" for cls, cnt in zip(unique, counts)])
        print(f"  Subject {sid}: Total={len(y)} ({len(epochs.ch_names)} ch) → {dist}")


def split_data(X, y, train_ratio=0.6, val_ratio=0.2, random_state=None):
    if X is None or y is None: 
        return None, None, None, None, None, None
    n = len(X); idx = np.arange(n)

    if n == 0: 
        return X, y, X, y, X, y
    if random_state is not None: 
        np.random.seed(random_state)

    np.random.shuffle(idx)
    train_end, val_end = int(train_ratio * n), int((train_ratio + val_ratio) * n)

    if not (0 < train_end < n and train_end < val_end <= n):
         logging.error(f"Invalid split indices: train_end={train_end}, val_end={val_end}, n={n}. Check ratios.");
         return None, None, None, None, None, None

    X_train, y_train = X[idx[:train_end]], y[idx[:train_end]]
    X_val, y_val = X[idx[train_end:val_end]], y[idx[train_end:val_end]]
    X_test, y_test = X[idx[val_end:]], y[idx[val_end:]]

    logging.info(f"Data split into: Train ({len(y_train)}), Validation ({len(y_val)}), Test ({len(y_test)})")
    return X_train, y_train, X_val, y_val, X_test, y_test
