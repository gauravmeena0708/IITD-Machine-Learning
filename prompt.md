    I want you to help me generate a modular deep learning project structure for a new supervised learning task (e.g., image classification, time series prediction, etc.). The structure should be easily reusable and configurable.
    
    I need the following:
    ✅ 1. Folder & File Structure:
    
        project_name/
        ├── config.py                  # Central settings
        ├── main.py                    # Main training script
        ├── optuna_search.py           # For hyperparameter tuning
        ├── test_pipeline.py           # For quick end-to-end check
        ├── models/
        │   ├── __init__.py
        │   ├── model1.py
        │   ├── model2.py
        ├── data/
        │   └── (placeholder for input files)
        ├── logs/
        └── results/
    
        ✅ 2. Each file should include:
        📁 config.py
    
            Dataset name/source
    
            Model name
    
            Dropout, L2, learning rate
    
            Augmentation flags
    
            Batch size, epochs, early stopping
    
            WANDB project name
    
        📁 data_loader.py
    
            Load data based on source name (e.g., "CIFAR10", "BCI_IV_2A")
    
            Support local CSV or built-in datasets
    
        📁 preprocessor.py
    
            Normalize, standardize
    
            One-hot encode labels
    
            Optionally augment (Gaussian noise or flips)
    
            Log class distributions
    
        📁 models/
    
            Each model as build_model(...) with input_shape, num_classes, **kwargs
    
            Central registry with get_model_by_name() and list_available_models()
    
        📁 trainer.py
    
            Train model with class weights
    
            Early stopping support
    
            Return history
    
        📁 evaluator.py
    
            Evaluate on validation or test
    
            Return accuracy, confusion matrix, and classification report
    
        📁 main.py
    
            Use config.py
    
            Load data → preprocess → train → evaluate → save model
    
        📁 optuna_search.py
    
            Search dropout, l2, learning rate, batch size
    
            Log all to wandb
    
            Save best .keras and JSON config
    
        📁 test_pipeline.py
    
            Load a few samples
    
            Build model
    
            Train for 3 epochs
    
            Run eval
    
        ✅ Additional Requirements:
    
            Use wandb and optuna if USE_WANDB = True flag in config
    
            Save best model as .keras
    
            Write all logs to logs/ and results/<experiment_tag>/
    
            Print class-wise distribution in each set
    
            Use input_shape + num_classes everywhere as standard model interface
    
            Easy to extend with new models or datasets
    
        🧪 First Example Task:
    
        Start by generating the full project structure for a multiclass image classification task using CIFAR-10, using 2 CNN models (cnn_simple, cnn_batchnorm). Use TensorFlow/Keras.
    
        Once you're done, I’ll ask you to adapt it for time series, EEG, or NLP tasks.
    
    🔁 You Can Reuse This Prompt For:
    
        EEG signal classification
    
        NLP sentiment detection
    
        Tabular fraud detection
    
        Regression tasks
    
    Just change the "First Example Task" section to fit your new domain.
