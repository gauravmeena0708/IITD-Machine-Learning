import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.compose import ColumnTransformer
import argparse

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

# Initialize stopwords and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Text preprocessing (tokenization, stopword removal, lemmatization)
def preprocess_text(text):
    tokens = word_tokenize(text) 
    processed_tokens = [lemmatizer.lemmatize(word.lower()) for word in tokens 
                        if word.lower() not in stop_words and word.isalpha()]
    return ' '.join(processed_tokens)

# Normalize numeric columns by their sum and handle missing values
def normalize_numeric_columns(df, numeric_columns):
    # Create a column for the sum of all numeric counts
    df['numeric_sum'] = df[numeric_columns].sum(axis=1)

    # Normalize each numeric column by the sum
    for col in numeric_columns:
        df[col + '_ratio'] = df[col] / df['numeric_sum']

    # Drop the sum column
    df.drop(columns=['numeric_sum'], inplace=True)

    # Return only the ratio columns
    ratio_columns = [col + '_ratio' for col in numeric_columns]
    
    # Fill NaN values in ratio columns with 0
    df[ratio_columns] = df[ratio_columns].fillna(0)
    return df, ratio_columns

# Function to load data
def load_data(file_path):
    df = pd.read_csv(file_path, sep='\t', header=None, usecols=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], 
                     names=['label', 'sentence', 'subject', 'speaker', 'job_title', 'state_info', 'party_affiliation',
                            'barely_true_counts', 'false_counts', 'half_true_counts', 'mostly_true_counts',
                            'pants_on_fire_counts', 'context'])
    return df

# Function to evaluate model and save results
def evaluate_model(model, X_valid, y_valid, label_encoder, out_file):
    y_pred = model.predict(X_valid)

    # Evaluate the model
    accuracy = accuracy_score(y_valid, y_pred)
    report = classification_report(y_valid, y_pred, target_names=label_encoder.classes_)

    # Print and save results
    print(f"Best Parameters: {model.best_params_}")
    print(f"Best Cross-validation Accuracy: {model.best_score_:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(report)

    with open(out_file, 'w') as f:
        f.write(f"Best Parameters: {model.best_params_}\n")
        f.write(f"Best Cross-validation Accuracy: {model.best_score_:.4f}\n")
        f.write(f"Test Accuracy: {accuracy:.4f}\n")
        f.write(report)

    print(f"Results saved to {out_file}")

# Main function
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, help="Train File Path")
    parser.add_argument('--test', type=str, help='Test File Path')
    parser.add_argument('--out', type=str, help='Output File Path for results')

    args = parser.parse_args()

    # Load train and validation data using the load_data function
    train_df = load_data(args.train)
    valid_df = load_data(args.test)

    # Define numeric columns to be normalized
    numeric_columns = ['barely_true_counts', 'false_counts', 'half_true_counts', 'mostly_true_counts', 'pants_on_fire_counts']

    # Normalize numeric columns for both train and validation data
    train_df, ratio_columns = normalize_numeric_columns(train_df, numeric_columns)
    valid_df, _ = normalize_numeric_columns(valid_df, numeric_columns)

    # Preprocess text data
    train_df['processed_text'] = train_df['sentence'].apply(preprocess_text)
    valid_df['processed_text'] = valid_df['sentence'].apply(preprocess_text)

    # Define categorical and numerical features
    categorical_features = ['subject', 'speaker', 'job_title', 'state_info', 'party_affiliation']
    numerical_features = ratio_columns 
    text_feature = 'processed_text'

    # ColumnTransformer for preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ('text', TfidfVectorizer(ngram_range=(1, 3), max_df=0.95, min_df=5), text_feature),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
            ('num', 'passthrough', numerical_features)
        ])

    # Pipeline with preprocessing, feature selection, and classifier
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('feature_selection', SelectKBest(chi2, k=100)),  # Select 100 best features
        ('clf', LogisticRegression(max_iter=1000))  
    ])

    # Define parameter grid for GridSearchCV (expanded)
    param_grid = {
        'clf__C': [0.1, 1, 10], 
        'clf__penalty': ['l1', 'l2'],  # Add more parameters for LogisticRegression 
    }

    # StratifiedKFold for cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # GridSearchCV for hyperparameter tuning
    grid_search = GridSearchCV(pipeline, param_grid=param_grid, cv=cv, scoring='accuracy')

    # Encode target labels
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_df['label'])
    y_valid = label_encoder.transform(valid_df['label'])

    # Fit the model using GridSearchCV
    grid_search.fit(train_df, y_train)

    # Evaluate the model and save results using the evaluate_model function
    evaluate_model(grid_search, valid_df, y_valid, label_encoder, args.out)