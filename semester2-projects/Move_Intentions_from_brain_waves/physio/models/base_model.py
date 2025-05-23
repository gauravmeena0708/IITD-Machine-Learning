# models/base_model.py

import tensorflow as tf
from tensorflow.keras.models import Model
import abc

class BaseEEGModel(abc.ABC):
    """
    Abstract base class for all EEG models.
    Ensures consistent build interface across models.
    """

    @staticmethod
    @abc.abstractmethod
    def build(input_shape, num_classes=2, **kwargs) -> Model:
        """
        Build and return a Keras EEG model.

        Args:
            input_shape (tuple): Shape of input EEG data (channels, time, 1)
            num_classes (int): Number of output classes
            **kwargs: Additional model-specific arguments

        Returns:
            tf.keras.Model: Compiled Keras model
        """
        pass
