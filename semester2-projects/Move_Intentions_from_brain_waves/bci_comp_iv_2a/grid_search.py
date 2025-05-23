# main_grid_search.py (Loading GDF, Using Multiple Models - CORRECTED)

# --- Essential Imports ---
import os
import numpy as np
import tensorflow as tf
import mne # For loading GDF and processing
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import itertools
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv2D, DepthwiseConv2D, SeparableConv2D,
                                     BatchNormalization, Activation, AveragePooling2D,
                                     Dropout, Flatten, Dense, Concatenate)
from tensorflow.keras.constraints import max_norm # Re-added for EEGNet_MSD
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.optimizers.schedules import LearningRateSchedule
from tensorflow.keras.callbacks import Callback, LearningRateScheduler, ModelCheckpoint, EarlyStopping # Added EarlyStopping
import time
import logging
import math
import traceback
# No scipy.io needed if loading GDF directly

# --- Import Model Building Functions ---
try:
    # Assuming eeg_models.py contains the necessary functions
    from eeg_models import (build_eegnet, build_eegnet_mha, build_eegnet_msd,
                           build_eegnet_msd_mha, build_q_eegnet, build_q_eegnet_mha)
    logging.info("Successfully imported model builder functions from eeg_models.py")
except ImportError as e:
    logging.error(f"Could not import model functions from eeg_models.py: {e}")
    exit()
except Exception as e: # Catch other potential import errors
     logging.error(f"An error occurred during model import: {e}")
     exit()

# --- Configuration ---
GDF_FILE_PATH = "/home/scai/mtech/aib242287/COL761/Assignment-3/dataset/A01T.gdf" # Source GDF file - PLEASE VERIFY THIS PATH
BASE_LOG_DIR = "eeg_multi_model_gdf_grid_search_logs" # Updated Log dir name
os.makedirs(BASE_LOG_DIR, exist_ok=True)

# Event mapping for Left (769) vs Right (770) hand
EVENT_ID_MAPPING = {'769': 7, '770': 8} # Using distinct integers > standard mne codes
INTERNAL_LABEL_MAP = {v: i for i, v in enumerate(sorted(EVENT_ID_MAPPING.values()))} # Maps 7->0, 8->1
NUM_CLASSES = len(EVENT_ID_MAPPING)
TARGET_NAMES = ['Left Hand (769)', 'Right Hand (770)']

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# --- Grid Search Parameter Lists ---
# Define which models to include in the grid search
model_builders = {
    # 'EEGNet': build_eegnet,
    # 'EEGNetMHA': build_eegnet_mha,
    # 'EEGNetMSD': build_eegnet_msd,
    # 'EEGNetMSDMHA': build_eegnet_msd_mha,
    'QEECNet': build_q_eegnet, # Reminder: Lighter variant, not true quantization
    # 'QEECNetMHA': build_q_eegnet_mha # Reminder: Lighter variant, not true quantization
}
# Hyperparameters (Expand these lists for a wider search)
optimizers_list = ['adam','sgd','rmsprop'] # Example: added adam
max_norm_vals_list = [1.0]    # Example: added another max norm
num_heads_list = [4]             # Relevant for MHA models

lrs_list = [0.001]
dropouts_list = [0.25, 0.4]
batch_sizes_list = [ 64]
schedulers_list = ['none', 'step', 'clr']
l2_rates_list = [1e-4, 1e-5] # Renamed from reg_rates_list
low_freq_list = [8.0, 5.0, 15.0]
high_freq_list = [35.0, 40.0]
tmin_list = [0.0, 0.5, 1.0]
tmax_list = [4.0, 3.0, 2.0]

# --- Training Configuration ---
EPOCHS_GRID = 500 # Fixed number of epochs

EARLY_STOPPING_PATIENCE = 500 # Patience for EarlyStopping

# --- Model Saving Configuration ---
PERIODIC_SAVE_START_EPOCH = 300  # Start saving periodically after epoch 50
PERIODIC_SAVE_MIN_ACC = 0.85 # Min val acc to save periodically
PERIODIC_SAVE_FREQ = 10      # Save every 10 epochs (if conditions met)

# --- Setup Logging ---
LOG_FILENAME = os.path.join(BASE_LOG_DIR, "grid_search_log.txt")
for handler in logging.root.handlers[:]: logging.root.removeHandler(handler)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler(LOG_FILENAME, mode='w'), logging.StreamHandler()])
logging.info("--- Starting Grid Search Script (Loading GDF, Multiple Models) ---")
logging.info(f"TensorFlow Version: {tf.__version__}")
logging.info(f"MNE Version: {mne.__version__}")
gpus = tf.config.list_physical_devices('GPU')
if gpus: logging.info(f"GPUs Available: {gpus}")
else: logging.info("No GPU detected by TensorFlow.")


# --- Custom Learning Rate Schedule (Triangular) ---
class CustomCyclicLR(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Custom Cyclic Learning Rate Schedule (Triangular)."""
    def __init__(self, base_lr, max_lr, step_size, name="CustomCyclicLR"):
        super().__init__()
        self.base_lr = tf.cast(base_lr, tf.float32)
        self.max_lr = tf.cast(max_lr, tf.float32)
        self.step_size = tf.cast(step_size, tf.float32)
        self.step_counter = tf.Variable(0., trainable=False, dtype=tf.float32, name="step_counter")
        self.name = name
    def __call__(self, step=None):
        # Use optimizer's iterations if available, otherwise internal counter
        current_step = tf.cast(step, tf.float32) if step is not None else self.step_counter
        if step is None:
            self.step_counter.assign_add(1.0)

        safe_step_size = tf.maximum(self.step_size, 1e-9)
        cycle = tf.floor(1 + current_step / (2 * safe_step_size))
        x = tf.abs(current_step / safe_step_size - 2 * cycle + 1)
        lr = self.base_lr + (self.max_lr - self.base_lr) * tf.maximum(0., (1 - x))
        return lr
    def get_config(self):
        return { "base_lr": float(self.base_lr.numpy()) if hasattr(self.base_lr, 'numpy') else float(self.base_lr),
                 "max_lr": float(self.max_lr.numpy()) if hasattr(self.max_lr, 'numpy') else float(self.max_lr),
                 "step_size": float(self.step_size.numpy()) if hasattr(self.step_size, 'numpy') else float(self.step_size),
                 "name": self.name }

# --- Custom Logging Callback ---
class EpochLogger(Callback):
    """Logs training/validation accuracy, loss, and LR at the end of each epoch."""
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current_lr = self.model.optimizer.learning_rate
        lr_val = -1.0
        try:
            if isinstance(current_lr, tf.keras.optimizers.schedules.LearningRateSchedule):
                # Use iterations which increments correctly per step (batch)
                step = self.model.optimizer.iterations
                lr_val = current_lr(step).numpy()
            elif hasattr(current_lr, 'numpy'): # Handle tf.Variable case
                lr_val = current_lr.numpy()
            else: # Handle float case
                lr_val = current_lr # Already a float
        except Exception as e:
            # logging.debug(f"Could not fetch LR: {e}") # Optional debug log
            pass # Ignore LR fetch errors during logging if needed
        logging.info(f"Epoch {epoch+1:03d}/{self.params['epochs']}: TrAcc={logs.get('accuracy', -1):.4f}, ValAcc={logs.get('val_accuracy', -1):.4f}, TrLoss={logs.get('loss', -1):.4f}, ValLoss={logs.get('val_loss', -1):.4f}, LR={lr_val:.6f}")


# --- Custom Callback for Periodic Model Saving ---
class PeriodicModelCheckpoint(Callback):
    """Saves the model weights periodically based on epoch and validation accuracy."""
    def __init__(self, base_filepath, start_epoch, min_val_accuracy, frequency):
        super().__init__()
        self.base_filepath = base_filepath
        self.start_epoch = start_epoch
        self.min_val_accuracy = min_val_accuracy
        self.frequency = frequency
        os.makedirs(self.base_filepath, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        current_epoch = epoch + 1
        if (current_epoch >= self.start_epoch):
            val_accuracy = logs.get('val_accuracy')
            if val_accuracy is not None and val_accuracy >= self.min_val_accuracy:
                filepath = os.path.join(self.base_filepath, f"eeg_{current_epoch:03d}_{val_accuracy:.4f}.weights.h5")
                # --- CORRECTED INDENTATION ---
                logging.info(f"Saving model checkpoint at epoch {current_epoch} to {filepath} (Val Acc >= {self.min_val_accuracy:.2f})")
                try:
                    self.model.save_weights(filepath, overwrite=True)
                except Exception as e:
                    logging.error(f"Error saving model checkpoint at epoch {current_epoch}: {e}")
                # --- END CORRECTION ---

# --- Custom Callback for Timing ---
class TimingCallback(Callback):
    """Records the time taken for each epoch and total training time."""
    def __init__(self):
        super().__init__()
        self.epoch_times = []
        self.epoch_start_time = 0
        self.train_start_time = 0
        self.train_end_time = 0
    def on_train_begin(self, logs=None):
        self.train_start_time = time.time()
        self.epoch_times = []
    def on_train_end(self, logs=None):
        self.train_end_time = time.time()
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()
    def on_epoch_end(self, epoch, logs=None):
        self.epoch_times.append(time.time() - self.epoch_start_time)
    def get_epoch_times(self): return self.epoch_times
    def get_total_time(self): return self.train_end_time - self.train_start_time if self.train_start_time and self.train_end_time else 0

# --- Data Preprocessing Function (Using GDF Loader Workaround) ---
def load_and_preprocess_gdf(gdf_file_path, low_freq, high_freq, event_id_mapping):
    """Loads GDF, applies filtering based on parameters, using a two-step load."""
    if not os.path.exists(gdf_file_path):
        raise FileNotFoundError(f"GDF file not found: {gdf_file_path}")
    raw = None
    try:
        logging.info(f"Loading GDF header/annotations: {gdf_file_path}...")
        # Load annotations first, but not data yet
        raw = mne.io.read_raw_gdf(gdf_file_path, preload=False, verbose='warning')
        logging.info("Header and annotations loaded.")
        sfreq = raw.info['sfreq']

        logging.info(f"Extracting events from annotations using mapping: {event_id_mapping}...")
        # Use the user-provided mapping { '769': 7, '770': 8 }
        events, event_dict_mne_internal = mne.events_from_annotations(raw, event_id=event_id_mapping, verbose='warning')
        # event_dict_mne_internal will likely be {'Left Hand (769)': 7, 'Right Hand (770)': 8}
        logging.info(f"Found {len(events)} events corresponding to specified annotations.")
        if len(events) == 0:
            raise ValueError("No relevant events found via annotations using the provided mapping.")

        logging.info("Loading GDF data into memory...")
        raw.load_data(verbose='warning')
        logging.info("Data loaded into memory.")

        # Pick only EEG channels (first 22 as per PDF description)
        try:
            # Explicitly pick the first 22 channels if they represent EEG
            num_eeg_channels = 22
            if len(raw.ch_names) >= num_eeg_channels:
                 picks = raw.ch_names[:num_eeg_channels]
                 raw.pick(picks=picks)
                 logging.info(f"Selected the first {num_eeg_channels} channels assumed to be EEG: {raw.ch_names}")
            else:
                 logging.warning(f"Found fewer than {num_eeg_channels} channels. Using all {len(raw.ch_names)} channels.")
                 raw.pick(picks='eeg', errors='ignore') # Fallback to picking by type 'eeg' if names aren't reliable

            # Exclude EOG (usually last 3 channels in this dataset)
            raw.drop_channels([ch for ch in raw.ch_names if 'EOG' in ch.upper()], on_missing='ignore')
            logging.info(f"Channels after picking EEG and dropping EOG (if found): {len(raw.ch_names)}")

        except Exception as e:
            logging.warning(f"Error picking/dropping channels: {e}. Proceeding with available channels.")

        logging.info(f"Applying band-pass filter ({low_freq:.1f}-{high_freq:.1f} Hz)...")
        raw.filter(low_freq, high_freq, fir_design='firwin', skip_by_annotation='edge', verbose='warning')

        # Pass the original event_id_mapping { '769': 7, '770': 8 } for epoching
        return raw, events, sfreq, event_id_mapping

    except ValueError as e:
        logging.error(f"Event extraction error: {e}.")
        if raw:
             logging.error(f"Available annotation descriptions: {np.unique(raw.annotations.description)}")
        return None, None, 0, None
    except Exception as e:
        logging.error(f"Error during GDF loading/preprocessing: {e}", exc_info=True)
        return None, None, 0, None

# --- Data Preparation Function ---
def prepare_data_from_gdf(raw, events, sfreq, tmin, tmax, event_id_map_for_epochs, internal_label_map):
    """Creates epochs, applies artifact rejection, and prepares data arrays."""
    if raw is None or events is None or sfreq <= 0: return None, None, 0

    duration = tmax - tmin
    expected_samples = int(math.ceil(duration * sfreq))
    logging.info(f"Creating epochs ({tmin:.1f}s to {tmax:.1f}s relative to events -> {expected_samples} samples)...")
    try:
        # Use the { '769': 7, '770': 8 } mapping here for creating epochs
        epochs = mne.Epochs(raw, events, event_id=event_id_map_for_epochs, tmin=tmin, tmax=tmax,
                           baseline=None, # No baseline correction in this window
                           preload=True, verbose='warning')
        logging.info(f"Initial epochs created: {len(epochs)}")
        if len(epochs) == 0: raise ValueError("Epoch object empty after creation.")

        # --- ADDED: Basic Artifact Rejection ---
        original_num_epochs = len(epochs)
        # Simple peak-to-peak rejection - adjust threshold as needed
        reject_criteria = dict(eeg=150e-6) # Reject epochs where any EEG channel exceeds 150 uV peak-to-peak
        try:
            epochs.drop_bad(reject=reject_criteria, verbose='warning')
            logging.info(f"Epochs after PTP rejection (> {reject_criteria['eeg']*1e6:.0f} uV): {len(epochs)} (dropped {original_num_epochs - len(epochs)})")
        except Exception as reject_err:
            logging.warning(f"Could not apply artifact rejection: {reject_err}. Using all created epochs.")

        if len(epochs) == 0:
            logging.error("No epochs remaining after artifact rejection.")
            return None, None, 0
        # --- End Artifact Rejection ---

    except Exception as e:
        logging.error(f"Epoch creation or rejection error: {e}")
        return None, None, 0

    actual_samples_per_epoch = epochs.get_data().shape[-1]
    if actual_samples_per_epoch != expected_samples:
        logging.warning(f"Actual samples ({actual_samples_per_epoch}) differs from expected ({expected_samples}). Using actual.")

    X = epochs.get_data(units='uV') # Get data in microvolts
    logging.info(f"Epoch data shape: {X.shape}")

    # Map the original MNE event IDs (7, 8) to internal labels (0, 1)
    y_integers = epochs.events[:, -1] # Get the event IDs (e.g., 7, 8)
    try:
        y = np.array([internal_label_map[event_id] for event_id in y_integers])
    except KeyError as e:
        logging.error(f"Label mapping error for event ID {e}. Check INTERNAL_LABEL_MAP: {internal_label_map}")
        return None, None, 0

    logging.info(f"Labels shape: {y.shape}, Unique internal labels: {np.unique(y)}, Distribution: {np.bincount(y)}")

    # Reshape for Conv2D input (add channel dimension)
    X = X[..., np.newaxis]
    logging.info(f"Reshaped X shape for model: {X.shape}")

    return X, y, actual_samples_per_epoch


# ==============================================================================
# --- Grid Search Section ---
# ==============================================================================
logging.info("--- Setting Up Grid Search (Multiple Models) ---")

# Create all combinations of parameters
combinations = list(itertools.product(
    model_builders.items(), # Tuple of (model_name, model_builder_func)
    optimizers_list, lrs_list, dropouts_list, batch_sizes_list,
    schedulers_list, l2_rates_list, max_norm_vals_list, # Added max_norm
    num_heads_list,
    low_freq_list, high_freq_list, tmin_list, tmax_list
))
logging.info(f"Total combinations to test: {len(combinations)}")
logging.info(f"Epochs per combination (max): {EPOCHS_GRID}")

results = []
grid_start_time = time.time()

logging.info(f"--- Starting Grid Search ---")
for i, ( (model_name, model_builder), # Unpack the model tuple
          opt_name, lr, drop, batch, sched, l2r, mn, num_h, # Added mn
          low_f, high_f, t_min, t_max
        ) in enumerate(combinations):

    # --- Combination Setup ---
    combo_start_time = time.time()
    # Collect parameters used in this specific combination
    combo_params = { 'model': model_name, 'opt': opt_name, 'lr': lr, 'dropout': drop, 'batch': batch,
                     'sched': sched, 'l2': l2r, 'max_norm': mn, 'num_heads': num_h,
                     'low_f': low_f, 'high_f': high_f, 'tmin': t_min, 'tmax': t_max }

    # Create a unique string identifier for this configuration
    config_str = (f"Model-{model_name}_Opt-{opt_name}_LR-{lr:.0e}_Drop-{drop:.2f}_Batch-{batch}_Sched-{sched}_"
                  f"L2-{l2r:.0e}_MN-{mn:.1f}_Heads-{num_h}_Filt-{low_f:.1f}-{high_f:.1f}_Time-{t_min:.1f}-{t_max:.1f}")
    config_str = config_str.replace('.', 'p').replace('/','-').replace(':', '') # Make filename safe

    logging.info(f"\n>>> Combination {i+1}/{len(combinations)}: {model_name} <<<")
    logging.info(f"Parameters: {combo_params}")

    # Setup directories for saving results for this combination
    config_log_dir = os.path.join(BASE_LOG_DIR, config_str)
    periodic_model_save_dir = os.path.join(config_log_dir, "periodic_models")
    best_model_save_dir = os.path.join(config_log_dir, "best_model")
    plot_save_path_base = os.path.join(config_log_dir, "plot_training")
    try:
        os.makedirs(periodic_model_save_dir, exist_ok=True)
        os.makedirs(best_model_save_dir, exist_ok=True)
    except OSError as e:
        logging.error(f"Could not create directories for {config_str}: {e}. Saving to base log dir.")
        # Fallback directories if specific one fails
        config_log_dir = BASE_LOG_DIR
        periodic_model_save_dir = os.path.join(config_log_dir, f"{config_str}_periodic")
        best_model_save_dir = os.path.join(config_log_dir, f"{config_str}_best")
        plot_save_path_base = os.path.join(config_log_dir, f"{config_str}_plot")
        os.makedirs(periodic_model_save_dir, exist_ok=True) # Try creating fallback dirs
        os.makedirs(best_model_save_dir, exist_ok=True)

    # --- Data Preparation ---
    logging.info("Preparing data for current configuration...")
    # Pass the original EVENT_ID_MAPPING { '769': 7, '770': 8 }
    raw_obj, events_array, sfreq, event_map_for_epochs = load_and_preprocess_gdf(
        GDF_FILE_PATH, low_f, high_f, EVENT_ID_MAPPING
    )
    if raw_obj is None:
        logging.error("Data loading/preprocessing failed. Skipping combination.")
        results.append({'combo_index': i+1, **combo_params, 'val_accuracy': -1.0, 'error': "Data Prep Failed"})
        continue

    # Pass event_map_for_epochs ({ '769': 7, '770': 8 }) and internal_label_map ({7:0, 8:1})
    X_data, y_data, actual_samples = prepare_data_from_gdf(
        raw_obj, events_array, sfreq, t_min, t_max, event_map_for_epochs, INTERNAL_LABEL_MAP
    )
    del raw_obj, events_array # Free memory
    if X_data is None or actual_samples == 0:
        logging.error("Epoch extraction or artifact rejection failed. Skipping combination.")
        results.append({'combo_index': i+1, **combo_params, 'val_accuracy': -1.0, 'error': "Epoch Extraction/Rejection Failed"})
        continue

    # Convert labels to categorical format for TF/Keras
    y_data_cat = tf.keras.utils.to_categorical(y_data, num_classes=NUM_CLASSES)

    # --- Split data ---
    logging.info("Splitting data into training and validation sets...")
    if X_data.shape[0] < 5: # Need enough samples for robust split and training
        logging.error(f"Cannot split data - only {X_data.shape[0]} samples available after preprocessing. Skipping.")
        results.append({'combo_index': i+1, **combo_params, 'val_accuracy': -1.0, 'error': "Insufficient Samples for Split"})
        del X_data, y_data, y_data_cat # Cleanup
        continue
    try:
        X_tr_c, X_val_c, y_tr_c, y_val_c = train_test_split(
            X_data, y_data_cat, test_size=0.2, random_state=RANDOM_SEED, stratify=y_data # Stratify by original integer labels
        )
        logging.info(f"Train shape: {X_tr_c.shape}, Val shape: {X_val_c.shape}")
        logging.info(f"Train label distribution (0/1): {np.sum(y_tr_c, axis=0)}")
        logging.info(f"Val label distribution (0/1): {np.sum(y_val_c, axis=0)}")
    except ValueError as e:
        logging.error(f"Error during train/test split (possibly too few samples per class): {e}. Skipping.")
        results.append({'combo_index': i+1, **combo_params, 'val_accuracy': -1.0, 'error': f"Split Failed: {e}"})
        del X_data, y_data, y_data_cat # Cleanup
        continue
    del X_data, y_data, y_data_cat # Free memory

    # --- Model & Training Setup ---
    tf.keras.backend.clear_session() # Clear previous models from memory

    # --- Callbacks ---
    callbacks = [EpochLogger()] # Log epoch details
    timing_callback = TimingCallback()
    callbacks.append(timing_callback) # Log timing

    # Periodic checkpoint callback
    periodic_ckpt_callback = PeriodicModelCheckpoint(
        base_filepath=periodic_model_save_dir, start_epoch=PERIODIC_SAVE_START_EPOCH,
        min_val_accuracy=PERIODIC_SAVE_MIN_ACC, frequency=PERIODIC_SAVE_FREQ
    )
    callbacks.append(periodic_ckpt_callback)

    # Best model checkpoint callback (saves only the best weights based on val_accuracy)
    best_model_filepath = os.path.join(best_model_save_dir, "best_model.weights.h5")
    best_model_checkpoint = ModelCheckpoint(
        filepath=best_model_filepath, monitor='val_accuracy', mode='max',
        save_best_only=True, save_weights_only=True, verbose=0 # Set verbose=1 for more info
    )
    callbacks.append(best_model_checkpoint)

    # Early stopping callback
    early_stopping = EarlyStopping(
        monitor='val_loss', # Stop based on validation loss
        patience=EARLY_STOPPING_PATIENCE, # Number of epochs with no improvement
        restore_best_weights=False, # Let ModelCheckpoint handle restoring best weights
        mode='min', # Stop when val_loss stops decreasing
        verbose=1
    )
    callbacks.append(early_stopping)

    # --- LR Schedule ---
    lr_to_use = lr # Default is fixed LR
    if sched == 'step':
        epochs_drop = max(1.0, float(EPOCHS_GRID // 3)) # Drop LR every 1/3 epochs
        logging.info(f"Using Step Decay LR: initial={lr:.0e}, drops every {epochs_drop:.0f} epochs.")
        def step_decay(epoch):
            initial_lrate = lr
            drop = 0.5
            epochs_drop_rate = epochs_drop
            lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop_rate))
            return lrate
        callbacks.append(LearningRateScheduler(step_decay, verbose=0))
        # Optimizer still starts with the initial LR passed to it
    elif sched == 'clr':
        # Calculate steps per epoch based on training data size and batch size
        steps_per_epoch = math.ceil(len(X_tr_c) / batch)
        if steps_per_epoch <= 0: steps_per_epoch = 1 # Avoid division by zero if data is tiny
        epochs_per_half_cycle = 8 # How many epochs for one ascent/descent
        clr_step_size = steps_per_epoch * epochs_per_half_cycle
        clr_base_lr = lr / 10.0 # Base LR is often 1/10th of max LR
        lr_to_use = CustomCyclicLR(base_lr=clr_base_lr, max_lr=lr, step_size=clr_step_size)
        logging.info(f"Using Custom Cyclic LR: base={clr_base_lr:.1e}, max={lr:.1e}, step_size={int(clr_step_size)} steps.")
    else: # sched == 'none' or unknown
        logging.info(f"Using fixed LR: {lr_to_use}")

    # --- Optimizer ---
    if opt_name.lower() == 'adam': optimizer = Adam(learning_rate=lr_to_use)
    elif opt_name.lower() == 'sgd': optimizer = SGD(learning_rate=lr_to_use, momentum=0.9)
    elif opt_name.lower() == 'rmsprop': optimizer = RMSprop(learning_rate=lr_to_use)
    else:
        logging.warning(f"Unknown optimizer '{opt_name}'. Defaulting to Adam.")
        optimizer = Adam(learning_rate=lr_to_use)

    # --- Model Instantiation: Call selected builder function ---
    current_input_shape = X_tr_c.shape[1:] # Shape is (Channels, Samples, 1)
    logging.info(f"Input shape for model: {current_input_shape}")
    try:
        # Call the function stored in model_builder (e.g., build_eegnet_msd)
        # Pass necessary parameters, kwargs catch extras like num_heads for models that don't use them
        model = model_builder(
            input_shape=current_input_shape,
            num_classes=NUM_CLASSES,
            dropout_rate=drop,
            l2_rate=l2r,
            max_norm_val=mn, # Pass max_norm (used by MSD models)
            num_heads=num_h    # Pass num_heads (used by MHA models)
        )
        logging.info(f"Instantiated model: {model.name}")
        # model.summary(print_fn=logging.info) # Optional: log model summary
    except Exception as model_init_error:
        logging.error(f"Error initializing model {model_name}: {model_init_error}. Skipping combination.", exc_info=True)
        results.append({'combo_index': i+1, **combo_params, 'val_accuracy': -1.0, 'error': f"Model Init Error: {model_init_error}"})
        continue # Skip to the next combination

    # --- Compile & Train ---
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    logging.info(f"Training {model.name} (Max Epochs: {EPOCHS_GRID}, Batch: {batch}, EarlyStopping Patience: {EARLY_STOPPING_PATIENCE})...")

    # Initialize result variables
    current_error = None; history = None; val_loss, val_acc = -1.0, -1.0; precision, recall, f1_score = -1.0, -1.0, -1.0; training_time = -1.0; epochs_ran = 0

    try:
        fit_start_time = time.time()
        history = model.fit(X_tr_c, y_tr_c,
                            epochs=EPOCHS_GRID,
                            batch_size=batch,
                            verbose=0, # Set to 1 or 2 for more detailed Keras output per epoch
                            validation_data=(X_val_c, y_val_c),
                            callbacks=callbacks, # Includes EpochLogger, Checkpoints, EarlyStopping, Timing
                            shuffle=True)
        training_time = timing_callback.get_total_time()
        epochs_ran = len(history.history.get('loss', [])) # Actual epochs run
        logging.info(f"Training completed in {training_time:.2f} seconds over {epochs_ran} epochs.")

        # Evaluate (load best weights saved by ModelCheckpoint)
        logging.info("Evaluating model on validation set using best weights...")
        best_weights_found = False
        if os.path.exists(best_model_filepath):
             try:
                 model.load_weights(best_model_filepath) # Load the best weights
                 val_loss, val_acc = model.evaluate(X_val_c, y_val_c, verbose=0)
                 logging.info(f"--> Best Weights Val Loss: {val_loss:.4f}, Best Weights Val Acc: {val_acc:.4f}")
                 best_weights_found = True
             except Exception as load_eval_error:
                 logging.warning(f"Could not load/evaluate best weights from {best_model_filepath}: {load_eval_error}. Evaluating final model state.")
                 # Fallback: evaluate the model in its final state if loading failed
                 val_loss, val_acc = model.evaluate(X_val_c, y_val_c, verbose=0)
                 logging.info(f"--> Final State Val Loss: {val_loss:.4f}, Final State Val Acc: {val_acc:.4f}")
        else:
             logging.warning(f"Best model weights file not found at {best_model_filepath}. Evaluating final model state.")
             # Evaluate the final state if no best weights file was saved (e.g., training stopped early)
             val_loss, val_acc = model.evaluate(X_val_c, y_val_c, verbose=0)
             logging.info(f"--> Final State Val Loss: {val_loss:.4f}, Final State Val Acc: {val_acc:.4f}")

        # Classification Report and Confusion Matrix
        y_pred_probs = model.predict(X_val_c, batch_size=batch)
        y_pred_labels = np.argmax(y_pred_probs, axis=1) # Predicted class indices (0 or 1)
        y_true_labels = np.argmax(y_val_c, axis=1)     # True class indices (0 or 1)

        logging.info(f"Classification Report ({'Best Weights' if best_weights_found else 'Final State'}):")
        try:
            # Generate classification report string and dictionary
            report_str = classification_report(y_true_labels, y_pred_labels, digits=4, target_names=TARGET_NAMES, zero_division=0)
            report_dict = classification_report(y_true_labels, y_pred_labels, output_dict=True, digits=4, target_names=TARGET_NAMES, zero_division=0)
            logging.info("\n" + report_str)

            # Extract weighted average metrics (useful for overall performance)
            precision = report_dict.get('weighted avg', {}).get('precision', -1.0)
            recall = report_dict.get('weighted avg', {}).get('recall', -1.0)
            f1_score = report_dict.get('weighted avg', {}).get('f1-score', -1.0)

            # Generate and log confusion matrix
            cm = confusion_matrix(y_true_labels, y_pred_labels)
            logging.info(f"Confusion Matrix:\n{cm}")

        except Exception as report_error:
            logging.warning(f"Classification report or confusion matrix generation failed: {report_error}")
            current_error = f"ReportGenError: {report_error}"

    except Exception as fit_eval_error:
        training_time = timing_callback.get_total_time() if 'timing_callback' in locals() else -1.0
        epochs_ran = len(history.history['loss']) if history and 'loss' in history.history else 0
        logging.error(f"An error occurred during training or evaluation: {fit_eval_error}", exc_info=True)
        val_acc, val_loss = -1.0, -1.0; precision, recall, f1_score = -1.0, -1.0, -1.0
        current_error = f"FitEvalError: {fit_eval_error}"

    # --- Plotting ---
    if history and epochs_ran > 0: # Only plot if training actually happened
        logging.info("Generating training history plots...")
        epoch_times_list = timing_callback.get_epoch_times()[:epochs_ran]
        # Pad with 0 for the start time at epoch 0
        cumulative_time = np.cumsum([0] + epoch_times_list)
        epoch_axis = range(epochs_ran) # Epochs are 0 to epochs_ran-1

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f'Run {i+1}: {model_name}\n{config_str}', fontsize=9) # Slightly smaller font

        # Accuracy Plot
        if 'accuracy' in history.history and 'val_accuracy' in history.history:
            axes[0].plot(epoch_axis, history.history['accuracy'], label='Train Acc')
            axes[0].plot(epoch_axis, history.history['val_accuracy'], label='Val Acc')
            axes[0].set_title('Accuracy vs. Epoch'); axes[0].set_xlabel('Epoch'); axes[0].legend(); axes[0].grid(True)
        else: axes[0].set_title('Accuracy Plot Unavailable'); axes[0].text(0.5, 0.5, 'No Data', ha='center', va='center')

        # Loss Plot
        if 'loss' in history.history and 'val_loss' in history.history:
            axes[1].plot(epoch_axis, history.history['loss'], label='Train Loss')
            axes[1].plot(epoch_axis, history.history['val_loss'], label='Val Loss')
            axes[1].set_title('Loss vs. Epoch'); axes[1].set_xlabel('Epoch'); axes[1].legend(); axes[1].grid(True)
        else: axes[1].set_title('Loss Plot Unavailable'); axes[1].text(0.5, 0.5, 'No Data', ha='center', va='center')

        # Cumulative Time Plot
        # Need epochs_ran + 1 points for cumulative time plot (includes time=0 at epoch 0)
        if len(cumulative_time) == epochs_ran + 1:
            axes[2].plot(range(epochs_ran + 1), cumulative_time, label='Cumulative Time (s)', marker='.')
            axes[2].set_title('Training Time vs. Epoch'); axes[2].set_xlabel('Epoch'); axes[2].set_ylabel('Time (s)'); axes[2].legend(); axes[2].grid(True)
        else:
            logging.warning(f"Time plotting skipped: length mismatch (Time points: {len(cumulative_time)}, Epochs run+1: {epochs_ran + 1}).")
            axes[2].set_title('Time Plot Unavailable'); axes[2].text(0.5, 0.5, 'Length Mismatch', ha='center', va='center')


        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
        plot_filename = f"{plot_save_path_base}.png"
        try:
            plt.savefig(plot_filename)
            logging.info(f"Saved training plots to {plot_filename}")
        except Exception as plot_err:
            logging.error(f"Error saving plot: {plot_err}")
        plt.close(fig) # Close the figure to free memory

    # --- Store Results ---
    # Use the best validation accuracy achieved during training for this combination
    results.append({
        'combo_index': i+1,
        **combo_params, # Store all hyperparameter settings
        'config_str': config_str, # Store the unique config string
        'actual_epochs': epochs_ran,
        'training_time_s': training_time,
        'val_accuracy': val_acc, # Best validation accuracy (from loaded weights or final)
        'val_loss': val_loss,     # Corresponding validation loss
        'precision_w': precision, # Weighted precision
        'recall_w': recall,       # Weighted recall
        'f1_score_w': f1_score,   # Weighted F1-score
        'error': current_error    # Store any error message encountered
    })
    logging.info(f"--- Finished Combination {i+1}/{len(combinations)} ---")

    # --- Cleanup for next iteration ---
    del X_tr_c, X_val_c, y_tr_c, y_val_c, model, optimizer, history, callbacks # Cleanup memory
    if 'lr_to_use' in locals() and isinstance(lr_to_use, CustomCyclicLR): del lr_to_use
    tf.keras.backend.clear_session() # Clear TF graph


# --- End of Grid Search Loop ---

# --- Display Top Results ---
grid_end_time = time.time()
total_grid_time = grid_end_time - grid_start_time
logging.info(f"\n--- Grid Search Complete ---")
logging.info(f"Total Grid Search Duration: {total_grid_time:.2f} seconds ({total_grid_time/60:.2f} minutes)")

if results:
    # Sort results by validation accuracy (highest first), handling None or NaN
    best_results = sorted(
        results,
        key=lambda x: x.get('val_accuracy', -1.0) if x.get('val_accuracy') is not None and not math.isnan(x.get('val_accuracy', -1.0)) else -1.0,
        reverse=True
    )

    logging.info(f"\n🏆 Top {min(5, len(best_results))} Configurations (Sorted by Best Validation Accuracy):")
    for rank, config in enumerate(best_results[:5]):
        log_str = f"\n--- Rank {rank+1} --- (Index: {config.get('combo_index', 'N/A')})\n"
        log_str += f"  Config String: {config.get('config_str', 'N/A')}\n"
        # Log key parameters and metrics
        log_str += f"  {'Model':<18}: {config.get('model', 'N/A')}\n"
        log_str += f"  {'Val Accuracy':<18}: {config.get('val_accuracy', -1.0):.4f}\n"
        log_str += f"  {'Val Loss':<18}: {config.get('val_loss', -1.0):.4f}\n"
        log_str += f"  {'F1 Score (W)':<18}: {config.get('f1_score_w', -1.0):.4f}\n"
        log_str += f"  {'Epochs Run':<18}: {config.get('actual_epochs', 0)}\n"
        log_str += f"  {'Train Time (s)':<18}: {config.get('training_time_s', 0.0):.2f}\n"
        # Log hyperparameters for easy comparison
        log_str += f"  {'Optimizer':<18}: {config.get('opt', 'N/A')}\n"
        log_str += f"  {'Learning Rate':<18}: {config.get('lr', -1.0):.0e}\n"
        log_str += f"  {'LR Scheduler':<18}: {config.get('sched', 'N/A')}\n"
        log_str += f"  {'Dropout':<18}: {config.get('dropout', -1.0):.2f}\n"
        log_str += f"  {'Batch Size':<18}: {config.get('batch', -1)}\n"
        log_str += f"  {'Max Norm':<18}: {config.get('max_norm', -1.0):.1f}\n"
        if 'MHA' in config.get('model', ''): log_str += f"  {'Num Heads':<18}: {config.get('num_heads', 'N/A')}\n"
        # Add other important parameters as needed
        if config.get('error'): log_str += f"  {'ERROR':<18}: {config.get('error')}\n"
        logging.info(log_str.strip())

    # Optionally save all results to a file (e.g., CSV)
    try:
        import pandas as pd
        results_df = pd.DataFrame(best_results)
        results_csv_path = os.path.join(BASE_LOG_DIR, "grid_search_results.csv")
        results_df.to_csv(results_csv_path, index=False)
        logging.info(f"Saved full results to {results_csv_path}")
    except ImportError:
        logging.warning("Pandas not installed. Skipping saving results to CSV.")
    except Exception as save_err:
        logging.error(f"Error saving results to CSV: {save_err}")

else:
    logging.warning("No results were generated from the grid search. Check logs for errors in the first combination.")

logging.info("✅ Full multi-model grid search script execution finished.")