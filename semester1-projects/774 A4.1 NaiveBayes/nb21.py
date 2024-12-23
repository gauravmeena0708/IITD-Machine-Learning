import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from scipy.sparse import hstack, csr_matrix
import argparse

# Download NLTK resources
nltk.download('stopwords')


# Initialize stopwords and stemmer
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()


def preprocess_text(text):
    tokens = text.split()
    processed_tokens = [ps.stem(word.lower()) for word in tokens if word.lower() not in stop_words and word.isalpha()]
    return ' '.join(processed_tokens)

def incorporate_features(train_df, valid_df):
    # Preprocess text data (uni-grams + bi-grams)
    train_df['processed_text'] = train_df['sentence'].apply(preprocess_text)
    valid_df['processed_text'] = valid_df['sentence'].apply(preprocess_text)

    # Initialize CountVectorizer for uni-grams and bi-grams
    vectorizer = CountVectorizer(ngram_range=(1, 2))
    X_train_text = vectorizer.fit_transform(train_df['processed_text'])
    X_valid_text = vectorizer.transform(valid_df['processed_text'])

    # Preprocess categorical features (subject, speaker, job title, etc.)
    categorical_columns = ['subject', 'speaker', 'job_title', 'state_info', 'party_affiliation', 'context']
    
    ohe = OneHotEncoder(handle_unknown='ignore')
    X_train_cat = ohe.fit_transform(train_df[categorical_columns])
    X_valid_cat = ohe.transform(valid_df[categorical_columns])

    # Process numeric features (credit history counts)
    numeric_columns = ['barely_true_counts', 'false_counts', 'half_true_counts', 'mostly_true_counts', 'pants_on_fire_counts']
    X_train_num = csr_matrix(train_df[numeric_columns].fillna(0))  # Fill missing values with 0
    X_valid_num = csr_matrix(valid_df[numeric_columns].fillna(0))  # Fill missing values with 0

    # Combine all features
    X_train_combined = hstack([X_train_text, X_train_cat, X_train_num])
    X_valid_combined = hstack([X_valid_text, X_valid_cat, X_valid_num])

    # Target labels
    y_train = train_df['label']
    y_valid = valid_df['label']

    # Train Multinomial Naive Bayes
    mnb = MultinomialNB()
    mnb.fit(X_train_combined, y_train)
    y_pred = mnb.predict(X_valid_combined)

    # Evaluate the model
    accuracy = accuracy_score(y_valid, y_pred)
    print(f"Validation Accuracy: {accuracy:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_valid, y_pred))

    return y_pred

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, help="Train File Path")
    parser.add_argument('--test', type=str, help='Test File Path')
    parser.add_argument('--out', type=str, help='Output File Path for model predictions')

    args = parser.parse_args()

    train_df = pd.read_csv(args.train, sep='\t', header=None, usecols=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], 
                           names=['label', 'sentence', 'subject', 'speaker', 'job_title', 'state_info', 'party_affiliation',
                                  'barely_true_counts', 'false_counts', 'half_true_counts', 'mostly_true_counts',
                                  'pants_on_fire_counts', 'context'])
    valid_df = pd.read_csv(args.test, sep='\t', header=None, usecols=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], 
                           names=['label', 'sentence', 'subject', 'speaker', 'job_title', 'state_info', 'party_affiliation',
                                  'barely_true_counts', 'false_counts', 'half_true_counts', 'mostly_true_counts',
                                  'pants_on_fire_counts', 'context'])

    # Call function to preprocess and train the model
    y_pred = incorporate_features(train_df, valid_df)

    # Save the predictions to the output file without headers or indices
    predictions_df = pd.DataFrame(y_pred)
    predictions_df.to_csv(args.out, index=False, header=False, sep='\n')
