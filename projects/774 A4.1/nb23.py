import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from scipy.sparse import hstack, csr_matrix
from sklearn.utils import class_weight
import numpy as np
import itertools
import xlsxwriter

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize stopwords and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    tokens = text.split()
    processed_tokens = [lemmatizer.lemmatize(word.lower()) for word in tokens if word.lower() not in stop_words and word.isalpha()]
    return ' '.join(processed_tokens)

def generate_feature_combinations(features):
    # Generate all combinations of features
    combinations = []
    for L in range(1, len(features) + 1):
        for subset in itertools.combinations(features, L):
            combinations.append(list(subset))
    return combinations

def evaluate_feature_combinations(train_df, valid_df, combinations):
    results = []

    # Preprocess text data (TF-IDF with uni-grams + bi-grams)
    train_df['processed_text'] = train_df['sentence'].apply(preprocess_text)
    valid_df['processed_text'] = valid_df['sentence'].apply(preprocess_text)

    # Initialize TfidfVectorizer for uni-grams and bi-grams
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_df=0.95)
    X_train_text = vectorizer.fit_transform(train_df['processed_text'])
    X_valid_text = vectorizer.transform(valid_df['processed_text'])

    # Preprocess categorical features (subject, speaker, job title, etc.)
    categorical_columns = ['subject', 'speaker', 'job_title', 'state_info', 'party_affiliation', 'context']
    ohe = OneHotEncoder(handle_unknown='ignore')
    X_train_cat = ohe.fit_transform(train_df[categorical_columns])
    X_valid_cat = ohe.transform(valid_df[categorical_columns])

    # Process numeric features (credit history counts)
    numeric_columns = ['barely_true_counts', 'false_counts', 'half_true_counts', 'mostly_true_counts', 'pants_on_fire_counts']
    scaler = StandardScaler(with_mean=False)
    X_train_num = scaler.fit_transform(train_df[numeric_columns].fillna(0))
    X_valid_num = scaler.transform(valid_df[numeric_columns].fillna(0))

    # Combine all features into a dictionary
    feature_dict = {
        'text': (X_train_text, X_valid_text),
        'categorical': (X_train_cat, X_valid_cat),
        'numeric': (csr_matrix(X_train_num), csr_matrix(X_valid_num))
    }

    # Encode target labels to numeric
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_df['label'])
    y_valid = label_encoder.transform(valid_df['label'])

    # Handle class imbalance with class weighting
    weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = {i: weights[i] for i in range(len(weights))}

    # Loop over each combination of features
    for combination in combinations:
        print(f"Evaluating combination: {combination}")

        # Concatenate features for the current combination
        X_train_combined = hstack([feature_dict[feat][0] for feat in combination])
        X_valid_combined = hstack([feature_dict[feat][1] for feat in combination])

        # Train Random Forest model with class weighting
        rf_clf = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight=class_weights, random_state=42)
        rf_clf.fit(X_train_combined, y_train)
        y_pred = rf_clf.predict(X_valid_combined)

        # Evaluate the model
        accuracy = accuracy_score(y_valid, y_pred)
        print(f"Accuracy for {combination}: {accuracy:.4f}")

        # Store results
        results.append({'Combination': '+'.join(combination), 'Accuracy': accuracy})

    return results

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, help="Train File Path")
    parser.add_argument('--test', type=str, help='Test File Path')
    parser.add_argument('--out', type=str, help='Output File Path for feature combination results')

    args = parser.parse_args()

    train_df = pd.read_csv(args.train, sep='\t', header=None, usecols=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], 
                           names=['label', 'sentence', 'subject', 'speaker', 'job_title', 'state_info', 'party_affiliation',
                                  'barely_true_counts', 'false_counts', 'half_true_counts', 'mostly_true_counts',
                                  'pants_on_fire_counts', 'context'])
    valid_df = pd.read_csv(args.test, sep='\t', header=None, usecols=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], 
                           names=['label', 'sentence', 'subject', 'speaker', 'job_title', 'state_info', 'party_affiliation',
                                  'barely_true_counts', 'false_counts', 'half_true_counts', 'mostly_true_counts',
                                  'pants_on_fire_counts', 'context'])

    # Define feature groups
    features = ['text', 'categorical', 'numeric']

    # Generate all feature combinations
    feature_combinations = generate_feature_combinations(features)

    # Evaluate each feature combination and collect results
    results = evaluate_feature_combinations(train_df, valid_df, feature_combinations)

    # Save results to Excel
    results_df = pd.DataFrame(results)
    results_df.to_excel(args.out, index=False, engine='xlsxwriter')

    print(f"Results saved to {args.out}")
