#config.py
import numpy as np

# -------------------- Dataset Configuration --------------------
DATASET_SOURCE = "EEGBCI"
DATASET_NAME = f"{DATASET_SOURCE}_MotorImagery"
AVAILABLE_SUBJECTS = list(range(1, 110))
SUBJECT_LIST = AVAILABLE_SUBJECTS
TRAIN_SUBJECT_PERCENT = 0.50
VAL_SUBJECT_PERCENT = 0.20
TEST_SUBJECT_PERCENT = 0.30

if not np.isclose(TRAIN_SUBJECT_PERCENT + VAL_SUBJECT_PERCENT + TEST_SUBJECT_PERCENT, 1.0):
    raise ValueError("Train, Validation, and Test subject percentages must sum to 1.0")

# -------------------- Event and Runs --------------------
if DATASET_SOURCE == "EEGBCI":
    RUNS = [3, 7, 11]
    EVENT_ID = dict(T1=1, T2=2)

NUM_CLASSES = len(EVENT_ID)
TMIN, TMAX = -0.5, 2.0
LOW_FREQ, HIGH_FREQ = 8., 35.
BASELINE_CORRECTION = (None, 0)
MONTAGE_NAME = 'standard_1005'

# -------------------- Augmentation Settings --------------------
APPLY_AUGMENTATION = True
NOISE_FACTOR = 0.03
NUM_AUG_COPIES = 1
RANDOM_SEED = 42
MODEL_NAME = "qeegnet_base"
DROPOUT_RATE = 0.5
L2_RATE = 1e-4

MODEL_KWARGS = dict(
    dropout_rate=DROPOUT_RATE,
    l2_rate=L2_RATE,
)

# -------------------- Training Configuration --------------------
EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 1e-2
EARLY_STOPPING_PATIENCE = 20
SAMPLING_FREQ =160
# -------------------- Experiment/Logging --------------------
WANDB_PROJECT = "subjects_multi_2504"
SHOW_PLOTS = False
SHOW_PER_SUBJECT_PLOTS = False
LOG_PLOTS_TO_WANDB = False
# -------------------- Derived Configs --------------------

EXPERIMENT_TAG = f"{MODEL_NAME}_dr{int(DROPOUT_RATE*100)}_l2e{int(np.log10(L2_RATE))}"
SAVE_DIR = f"../results/{EXPERIMENT_TAG}/"
N_CHANNELS = 64
N_TIMESTEPS = 400
CONFIG_DICT = {
    "dataset": DATASET_NAME,
    "subjects": len(AVAILABLE_SUBJECTS),
    "runs": RUNS,
    "event_id": EVENT_ID,
    "preprocessing": {
        "tmin": TMIN, "tmax": TMAX,
        "low_freq": LOW_FREQ, "high_freq": HIGH_FREQ,
        "baseline": BASELINE_CORRECTION,
        "montage": MONTAGE_NAME
    },
    "augmentation": {
        "apply": APPLY_AUGMENTATION,
        "noise_factor": NOISE_FACTOR,
        "num_copies": NUM_AUG_COPIES
    },
    "model": {
        "name": MODEL_NAME,
        "kwargs": MODEL_KWARGS
    },
    "training": {
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "early_stopping": EARLY_STOPPING_PATIENCE
    },
    "wandb": WANDB_PROJECT,
    "seed": RANDOM_SEED,
    "tag": EXPERIMENT_TAG
}
MODEL_LIST = [
    "eegnet_base", "eegnet_attn", "qeegnet_base", "qeegnet_attn",
    "msd_base", "msd_attn"
]