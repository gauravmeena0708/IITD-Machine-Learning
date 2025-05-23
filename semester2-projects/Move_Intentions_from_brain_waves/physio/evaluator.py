#evaluator.py
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import logging
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def evaluate_model(model, X_test, y_test_oh, y_test, target_names=None, subject_id=None):
    start = time.time()
    predictions = model.predict(X_test)
    duration = time.time() - start

    pred_labels = np.argmax(predictions, axis=1)
    acc = accuracy_score(y_test, pred_labels)

    msg = f"Test Accuracy: {acc:.4f}"
    if subject_id is not None:
        msg = f"[Subject {subject_id}] " + msg
    logging.info(msg)
    logging.info(f"Evaluation prediction time: {duration:.2f} seconds")


    if target_names:
        report = classification_report(y_test, pred_labels, target_names=target_names)
        cm = confusion_matrix(y_test, pred_labels)
    else:
        report, cm = None, None
    return acc, report, cm