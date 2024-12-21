import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import argparse

# Download NLTK resources
nltk.download('stopwords')


# Initialize stopwords and stemmer
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

# Preprocessing Function
def preprocess_text(text):
    tokens = text.split()
    processed_tokens = [ps.stem(word.lower()) for word in tokens if word.lower() not in stop_words and word.isalpha()]
    return ' '.join(processed_tokens)

# Function to Train and Evaluate Naive Bayes Model with Uni-grams and Bi-grams
def multinomial_nb_model(train_df, valid_df):
    # Apply preprocessing
    train_df['processed_text'] = train_df['sentence'].apply(preprocess_text)
    valid_df['processed_text'] = valid_df['sentence'].apply(preprocess_text)

    # Initialize CountVectorizer with both uni-grams and bi-grams
    vectorizer = CountVectorizer(ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(train_df['processed_text'])
    X_valid = vectorizer.transform(valid_df['processed_text'])

    y_train = train_df['label']
    y_valid = valid_df['label']

    print("Number of Features (Uni-grams + Bi-grams):", X_train.shape[1])

    # Train Multinomial Naive Bayes
    mnb = MultinomialNB()
    mnb.fit(X_train, y_train)
    y_pred = mnb.predict(X_valid)

    # Evaluate the model
    accuracy = accuracy_score(y_valid, y_pred)
    print(f"Validation Accuracy: {accuracy:.4f}")
    
    return y_pred

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, help="Train File Path")
    parser.add_argument('--test', type=str, help='Test File Path')
    parser.add_argument('--out', type=str, help='Output File Path for model predictions')

    args = parser.parse_args()

    # Load the datasets
    train_df = pd.read_csv(args.train, sep='\t', header=None, usecols=[1, 2], names=['label', 'sentence'])
    valid_df = pd.read_csv(args.test, sep='\t', header=None, usecols=[1, 2], names=['label', 'sentence'])

    print("Training Data Shape:", train_df.shape)
    print("Validation Data Shape:", valid_df.shape)

    # Get predictions from the model using uni-grams and bi-grams
    y_pred = multinomial_nb_model(train_df, valid_df)

    # Save the predictions to the output file without headers or indices
    predictions_df = pd.DataFrame(y_pred)
    predictions_df.to_csv(args.out, index=False, header=False, sep='\n')
