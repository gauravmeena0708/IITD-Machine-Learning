# eeg_classifier/data_loader.py

import os
import logging
import mne
import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_raw_data(subject: int, runs: list[int]):
    """
    Loads raw EEG data for a given subject depending on the configured dataset source.

    Args:
        subject (int): Subject ID.
        runs (list[int]): EEGBCI run numbers (ignored for BCI_IV_2A).

    Returns:
        mne.io.Raw or None
    """
    #logging.info(f"Loading Subject {subject} from {config.DATASET_SOURCE}...")

    try:
        if config.DATASET_SOURCE == "EEGBCI":
            raw_fnames = mne.datasets.eegbci.load_data(subject, runs, update_path=True, verbose=False)
            raws = [mne.io.read_raw_edf(f, preload=True, verbose=False) for f in raw_fnames]
            raw = mne.concatenate_raws(raws, verbose=False)
            return raw

        elif config.DATASET_SOURCE == "BCI_IV_2A":
            subject_str = f"{subject:02d}"
            gdf_path = os.path.join("data", "BCI_IV_2A", f"A0{subject_str}T.gdf")
            if not os.path.isfile(gdf_path):
                raise FileNotFoundError(f"GDF file not found: {gdf_path}")
            raw = mne.io.read_raw_gdf(gdf_path, preload=True, verbose=False)
            return raw

        else:
            raise ValueError(f"Unsupported dataset source: {config.DATASET_SOURCE}")

    except Exception as e:
        logging.error(f"Failed to load subject {subject}: {e}", exc_info=True)
        return None

# data_loader.py

import mne
import config
import numpy as np

def load_data_for_subject(subject, runs=config.RUNS, event_id=config.EVENT_ID):
    raw_fnames = mne.datasets.eegbci.load_data(subject, runs)
    raw = mne.io.read_raw_edf(raw_fnames[0], preload=True)
    for f in raw_fnames[1:]:
        raw.append(mne.io.read_raw_edf(f, preload=True))

    raw.filter(config.LOW_FREQ, config.HIGH_FREQ)
    raw.set_montage(config.MONTAGE_NAME, on_missing='ignore')

    events, _ = mne.events_from_annotations(raw)
    picks = mne.pick_types(raw.info, eeg=True, exclude="bads")

    epochs = mne.Epochs(raw, events, event_id, config.TMIN, config.TMAX,
                        proj=True, picks=picks, baseline=config.BASELINE_CORRECTION,
                        preload=True)

    labels = epochs.events[:, -1] - 1  # convert 1,2 -> 0,1
    data = epochs.get_data()  # shape: (n_epochs, n_channels, n_times)

    # Reshape for CNN input: (samples, channels, time, 1)
    data = data[..., np.newaxis]

    return data, labels

