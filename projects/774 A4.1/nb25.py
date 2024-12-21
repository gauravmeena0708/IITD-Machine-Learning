import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
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
    
    # Fill NaN values in ratio columns with 0 (can be changed to mean if appropriate)
    df[ratio_columns] = df[ratio_columns].fillna(0)  # Filling NaN with 0
    return df, ratio_columns

def generate_ratio_combinations(ratio_columns):
    # Generate all combinations of ratio columns
    ratio_combinations = []
    for L in range(1, len(ratio_columns) + 1):
        for subset in itertools.combinations(ratio_columns, L):
            ratio_combinations.append(list(subset))
    return ratio_combinations

def evaluate_feature_combinations(train_df, valid_df, ratio_combinations):
    results = []

    # Preprocess text data (TF-IDF with uni-grams + bi-grams)
    train_df['processed_text'] = train_df['sentence'].apply(preprocess_text)
    valid_df['processed_text'] = valid_df['sentence'].apply(preprocess_text)

    # Initialize TfidfVectorizer for uni-grams and bi-grams
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_df=0.95)
    X_train_text = vectorizer.fit_transform(train_df['processed_text'])
    X_valid_text = vectorizer.transform(valid_df['processed_text'])

    # Preprocess categorical features (subject, speaker, state_info, party_affiliation, context)
    categorical_columns = ['subject', 'speaker', 'state_info', 'party_affiliation', 'context']
    ohe = OneHotEncoder(handle_unknown='ignore')

    categorical_features = {}
    for cat in categorical_columns:
        X_train_cat = ohe.fit_transform(train_df[[cat]])
        X_valid_cat = ohe.transform(valid_df[[cat]])
        categorical_features[cat] = (X_train_cat, X_valid_cat)

    # Combine all constant features into a dictionary
    feature_dict = {
        'text': (X_train_text, X_valid_text),
    }
    for cat in categorical_columns:
        feature_dict[cat] = categorical_features[cat]

    # Encode target labels to numeric
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_df['label'])
    y_valid = label_encoder.transform(valid_df['label'])

    # Handle class imbalance with class weighting
    weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = {i: weights[i] for i in range(len(weights))}

    # Loop over each combination of numeric ratios
    for ratio_combination in ratio_combinations:
        print(f"Evaluating combination: text + categorical + {ratio_combination}")

        # Combine constant features (text + categorical) with the ratio columns
        combined_features = [feature_dict['text'][0]] + [feature_dict[cat][0] for cat in categorical_columns]
        combined_features_valid = [feature_dict['text'][1]] + [feature_dict[cat][1] for cat in categorical_columns]

        # Add the ratio columns to the feature set
        X_train_ratios = csr_matrix(train_df[ratio_combination].values)
        X_valid_ratios = csr_matrix(valid_df[ratio_combination].values)

        combined_features.append(X_train_ratios)
        combined_features_valid.append(X_valid_ratios)

        # Concatenate all the features
        X_train_combined = hstack(combined_features)
        X_valid_combined = hstack(combined_features_valid)

        # Train Random Forest model with class weighting
        rf_clf = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight=class_weights, random_state=42)
        rf_clf.fit(X_train_combined, y_train)
        y_pred = rf_clf.predict(X_valid_combined)

        # Evaluate the model
        accuracy = accuracy_score(y_valid, y_pred)
        print(f"Accuracy for text + categorical + {ratio_combination}: {accuracy:.4f}")

        # Store results
        results.append({'Combination': 'text+categorical+' + '+'.join(ratio_combination), 'Accuracy': accuracy})

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

    # Define numeric columns to be summed and normalized
    numeric_columns = ['barely_true_counts', 'false_counts', 'half_true_counts', 'mostly_true_counts', 'pants_on_fire_counts']

    # Normalize numeric columns and generate ratio columns
    train_df, ratio_columns = normalize_numeric_columns(train_df, numeric_columns)
    valid_df, _ = normalize_numeric_columns(valid_df, numeric_columns)

    # Generate all possible combinations of the ratio columns
    ratio_combinations = generate_ratio_combinations(ratio_columns)

    # Evaluate each feature combination and collect results
    results = evaluate_feature_combinations(train_df, valid_df, ratio_combinations)

    # Save results to Excel
    results_df = pd.DataFrame(results)
    results_df.to_excel(args.out, index=False, engine='xlsxwriter')

    print(f"Results saved to {args.out}")
