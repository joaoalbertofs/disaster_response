import sys

# Import libraries
import pandas as pd
from sqlalchemy import create_engine
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer, classification_report
import numpy as np
import pickle
import warnings

warnings.filterwarnings("ignore")

import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

def load_data(database_filepath):
    """
    Load data from a SQLite database.

    Args:
        database_filepath (str): Path to the SQLite database file.

    Returns:
        tuple: A tuple containing features, labels, and category names.
    """
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('DisasterResponse', engine)
    
    X = df['message']
    y = df.iloc[:, 4:]
    category_names = y.columns
    
    return X, y, category_names

def tokenize(text, url_place_holder_string="urlplaceholder"):
    """
    Tokenize and lemmatize text data.

    Args:
        text (str): Input text to be tokenized.
        url_place_holder_string (str): String to replace URLs with.

    Returns:
        list: List of cleaned and lemmatized tokens.
    """
    # Replace all URLs with a URL placeholder string
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for detected_url in detected_urls:
        text = text.replace(detected_url, url_place_holder_string)

    # Extract word tokens from the provided text
    tokens = nltk.word_tokenize(text)
    
    # Lemmatizer to remove inflectional and derivationally related forms of a word
    lemmatizer = nltk.WordNetLemmatizer()

    # List of clean tokens
    clean_tokens = [lemmatizer.lemmatize(w).lower().strip() for w in tokens]
    return clean_tokens

def build_model():
    """
    Build a machine learning model pipeline.

    Returns:
        sklearn.model_selection._search.GridSearchCV: Grid search model pipeline.
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters = {
        'tfidf__smooth_idf': [True],
        'clf__estimator__n_estimators': [40],
    }
    
    model = GridSearchCV(pipeline, parameters)
    
    return model

def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate the performance of the machine learning model.

    Args:
        model (sklearn.model_selection._search.GridSearchCV): Trained machine learning model.
        X_test (pandas.Series): Test features.
        Y_test (pandas.DataFrame): Test labels.
        category_names (list): List of category names.
    """
    y_pred = model.predict(X_test)
    
    for i, column in enumerate(category_names):
        print(f"Category: {column}\n")
        print(classification_report(Y_test[column], y_pred[:, i]))
        print("-" * 60)

def save_model(model, model_filepath):
    """
    Save the trained model to a pickle file.

    Args:
        model (sklearn.model_selection._search.GridSearchCV): Trained machine learning model.
        model_filepath (str): Path to save the pickle file.
    """
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()