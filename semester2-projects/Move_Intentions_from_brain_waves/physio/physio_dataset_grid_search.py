#physio_grid_search.py
import logging
import json
import csv
import itertools
import numpy as np
import os
import config
import preprocessor
import trainer
from models import get_model_by_name
import tensorflow as tf
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 
results_project_name = getattr(config, 'WANDB_PROJECT', 'eeg_grid_search')
RESULTS_DIR = os.path.join(BASE_DIR, "results", f"grid_{results_project_name}")
os.makedirs(RESULTS_DIR, exist_ok=True)

grid_log = os.path.join(RESULTS_DIR, "grid_search.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(grid_log, encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# Initialize CSV results file
CSV_PATH = os.path.join(RESULTS_DIR, "grid_search_results.csv")
if not os.path.isfile(CSV_PATH):
    try:
        with open(CSV_PATH, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "model_name", "trial_index",
                "dropout_rate", "l2_rate",
                "learning_rate", "batch_size",
                "optimizer", "scheduler",
                "val_accuracy", "test_accuracy"
            ])
        logging.info(f"Created result CSV at {CSV_PATH}")
    except IOError as e:
        logging.error(f"Failed to create CSV file at {CSV_PATH}: {e}")
        exit()


H_PARAM_GRID = {
    "dropout_rate": [0.3, 0.4, 0.5, 0.6],
    "l2_rate":      [1e-5, 1e-4, 1e-3, 1e-2],
    "learning_rate": [1e-4, 1e-3, 1e-2],
    "batch_size":   [16, 32, 64],
    "optimizer":    ["adam", "sgd"],
    "scheduler":    ["none", "plateau", "cosine", "step", "cyclic"],
}

def evaluate_model_with_config(model_name, trial_params):
    """Loads data, builds, trains, and evaluates a model for one hyperparameter combination."""
    try:
        subjects = getattr(config, 'AVAILABLE_SUBJECTS', list(range(1, 11))) 
        if not subjects:
            logging.error("No subjects defined in config.AVAILABLE_SUBJECTS")
            return None, 0.0, 0.0
        logging.info(f"Using {len(subjects)} subjects for this trial.")
        train_subj, val_subj, test_subj = preprocessor.split_subjects(
            subjects,
            config.TRAIN_SUBJECT_PERCENT,
            config.VAL_SUBJECT_PERCENT,
            config.TEST_SUBJECT_PERCENT,
            config.RANDOM_SEED
        )
        logging.info(f"Subject split: Train={len(train_subj)}, Val={len(val_subj)}, Test={len(test_subj)}")
        X_train, y_train, _ = preprocessor.load_and_aggregate_data(train_subj)
        X_val,   y_val,   _ = preprocessor.load_and_aggregate_data(val_subj)
        X_test,  y_test,  _ = preprocessor.load_and_aggregate_data(test_subj)

        if X_train is None or X_val is None or X_test is None:
             logging.error("Data loading failed for one or more splits.")
             return None, 0.0, 0.0
        if X_train.size == 0 or X_val.size == 0 or X_test.size == 0:
             logging.error("One or more data splits are empty after loading.")
             return None, 0.0, 0.0

        # --- Preprocessing ---
        y_train_oh, y_val_oh, y_test_oh = preprocessor.one_hot_encode_labels(
            config.NUM_CLASSES, y_train, y_val, y_test
        )
        class_weights = trainer.calculate_class_weights(y_train) 

        # --- Model Building ---
        tf.keras.backend.clear_session() 
        model_fn = get_model_by_name(model_name) 
        if model_fn is None:
             logging.error(f"Could not find model function for '{model_name}'")
             return None, 0.0, 0.0


        model_build_params = {
            'input_shape': X_train.shape[1:],
            'num_classes': config.NUM_CLASSES,
        }
        if 'dropout_rate' in trial_params: model_build_params['dropout_rate'] = trial_params['dropout_rate']
        if 'l2_rate' in trial_params: model_build_params['l2_rate'] = trial_params['l2_rate']


        model = model_fn(**model_build_params)
        if model is None:
            logging.error(f"Model building failed for '{model_name}'")
            return None, 0.0, 0.0

        # --- Model Training ---
        model, history = trainer.train_model( 
            model,
            X_train, y_train_oh,
            X_val,   y_val_oh,
            epochs=config.EPOCHS,
            batch_size=trial_params["batch_size"],
            learning_rate=trial_params["learning_rate"],
            class_weights=class_weights,
            early_stopping_patience=config.EARLY_STOPPING_PATIENCE, 
            optimizer_name=trial_params["optimizer"],
            scheduler_name=trial_params["scheduler"]
        )
        if model is None or history is None:
            logging.error(f"Model training failed for '{model_name}' trial.")
            return None, 0.0, 0.0

        logging.info("Evaluating trained model...")
        val_loss, val_acc = model.evaluate(X_val, y_val_oh, verbose=0)
        test_loss, test_acc = model.evaluate(X_test, y_test_oh, verbose=0)
        logging.info(f"Evaluation complete: Val Acc={val_acc:.4f}, Test Acc={test_acc:.4f}")

        return model, val_acc, test_acc

    except Exception as e:
        logging.error(f"Error during trial evaluation for {model_name} with params {trial_params}: {e}", exc_info=True)
        return None, 0.0, 0.0


def run_grid_search():
    """Runs the grid search over specified models and hyperparameters."""
    if not hasattr(config, 'MODEL_LIST') or not config.MODEL_LIST:
        logging.error("config.MODEL_LIST is not defined or empty. Cannot run grid search.")
        return

    logging.info(f"Starting grid search for models: {config.MODEL_LIST}")
    logging.info(f"Hyperparameter grid: {H_PARAM_GRID}")

    for model_name in config.MODEL_LIST:
        logging.info(f"\n--- Starting Grid Search for Model: {model_name} ---")
        trial_idx = 0
        param_keys = H_PARAM_GRID.keys()
        param_values = H_PARAM_GRID.values()
        param_combinations = list(itertools.product(*param_values))
        total_combinations = len(param_combinations)

        for combo in param_combinations:
            trial_idx += 1
            trial_params = dict(zip(param_keys, combo))
            logging.info(f"\n--- Running Trial {trial_idx}/{total_combinations} for {model_name} ---")
            logging.info(f"Parameters: {trial_params}")
            model, val_acc, test_acc = evaluate_model_with_config(model_name, trial_params)
            if model is not None:
                try:
                    model_filename = f"{model_name}_trial{trial_idx:03d}_val{val_acc:.4f}_test{test_acc:.4f}.keras"
                    model_path = os.path.join(RESULTS_DIR, model_filename)
                    model.save(model_path)
                    logging.info(f"Saved model to {model_path}")
                    params_to_save = {**trial_params, 'model_name': model_name, 'val_accuracy': val_acc, 'test_accuracy': test_acc}
                    json_path = model_path.replace('.keras', '.json')
                    with open(json_path, 'w') as f:
                        json.dump(params_to_save, f, indent=2)
                    logging.info(f"Saved trial params to {json_path}")
                    with open(CSV_PATH, "a", newline="") as f:
                        csv.writer(f).writerow([
                            model_name, trial_idx,
                            trial_params.get('dropout_rate', 'N/A'),
                            trial_params.get('l2_rate', 'N/A'),
                            trial_params.get('learning_rate', 'N/A'),
                            trial_params.get('batch_size', 'N/A'),
                            trial_params.get('optimizer', 'N/A'),
                            trial_params.get('scheduler', 'N/A'),
                            f"{val_acc:.4f}",
                            f"{test_acc:.4f}"
                        ])
                    logging.info(f"Appended results to {CSV_PATH}")

                except Exception as e:
                    logging.error(f"Error saving results for trial {trial_idx} of {model_name}: {e}", exc_info=True)
            else:
                 logging.warning(f"Skipping result saving for failed trial {trial_idx} of {model_name}.")
        logging.info(f"--- Finished Grid Search for Model: {model_name} ---")

    logging.info("--- Grid Search Completed ---")

if __name__ == '__main__':
    required_configs = ['MODEL_LIST', 'AVAILABLE_SUBJECTS', 'NUM_CLASSES',
                        'TRAIN_SUBJECT_PERCENT', 'VAL_SUBJECT_PERCENT', 'TEST_SUBJECT_PERCENT',
                        'RANDOM_SEED', 'EPOCHS', 'EARLY_STOPPING_PATIENCE']
    missing_configs = [attr for attr in required_configs if not hasattr(config, attr)]
    if missing_configs:
        logging.error(f"Missing required configurations in config.py: {missing_configs}")
    else:
        run_grid_search()

