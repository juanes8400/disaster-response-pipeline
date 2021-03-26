import sys
import re
import pickle
import nltk
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.ensemble import ExtraTreesClassifier
import joblib

def load_data(database_filepath):
    '''
    Input:
    database_filepath: The route of the file which contains the databases
    
    Output:
    X: The dataframe containing the exogenous variables for the model
    Y: Dataframe containing the endogenous (y) variable for the model
    category_names: The columns of the Y, taken from the column names
    
    This function load data from the constructed database and gets X, Y and category names.
    '''
    # load data from database
    engine = create_engine('sqlite:///messages.db')
    df = pd.read_sql("SELECT * FROM messages", engine)
    X = df.loc[:,['message']]
    Y = df.drop(['message', 'genre', 'id', 'original'], axis=1)
    category_names = Y.columns.tolist()
    return X, Y, category_names
    

def tokenize(text):
    '''
    Input:
    text: The complete text from the twits which are going to be analized
    Output:
    lems: The words preprocessed for the algorithm
    
    The function applies removal of stop words, lemmatization, removal of punctuation marks and lemmatization of the text, ready to be modeled.
    '''
    import string
    stop_words = nltk.corpus.stopwords.words("english")
    lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()
    remove_punc_table = str.maketrans('', '', string.punctuation)

    # Remove dots, commas, and other and upper cases
    text = text.translate(remove_punc_table).lower()
    
    # tokenization
    tokens = nltk.word_tokenize(text)
    
    # lemmatize and remove stop words
    lems = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    
    return lems


def build_model():
    
    '''
    Input: None
    Output: Model object ready to receive text
    This function creates the object model which pipelines and gridsearches the df of texts
    '''
    
    forest_clf = RandomForestClassifier(n_estimators=3,verbose=100)
    pipeline = Pipeline([
                    ('tfidf', TfidfVectorizer(tokenizer=tokenize)),
                    ('forest', MultiOutputClassifier(forest_clf))
                    ])
    parameters = {
        'tfidf__ngram_range': ((1, 1), (1, 2)),
        'tfidf__max_df': (0.8, 0.9, 1),
        'tfidf__max_features': (None, 6000),
        'forest__estimator__n_estimators': [10, 15, 20],
        'forest__estimator__min_samples_split': [2, 4, 6]}

    grid = GridSearchCV(pipeline, parameters, cv=2, n_jobs=-1)
    return(grid)

def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Input:
    Model: Model object to be inputted with the X variables
    X_test: Dataframe containing the test set of columns which will help model
    Y_test: Dataframe with the endogenous variables to evaluate (test)
    category_names: Category names for the Y
    
    Output:
    classification_report: category by category the confusion matrix and its analysis
    
    This function prints and save the model evaluation
    
    '''
    Y_pred = model.predict(X_test)
    for i, col in enumerate(Y_test):
        print(col)
        print(classification_report(Y_test[col], Y_pred[:, i]))
    return(classification_report(Y_test[col], Y_pred[:, i]))


def save_model(model, model_filepath):
    '''
    Input:
    model: Model object
    model_filepath: File path where the user would like to save the model
    
    Output:
    None
    
    The function saves "model" in the indicated path
    '''
    joblib.dump(model, '/home/workspace/models/classifier.pkl')


def main():
    '''
    Input: None
    Output: None
    
    This function runs all
    '''
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X.iloc[:,0], Y, test_size=0.2)
        
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