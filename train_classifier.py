# import libraries
import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import re
# NLP libraries
import nltk
nltk.download(['punkt', 'wordnet'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
#ML Libraries
from sklearn.pipeline import Pipeline
#from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
#Warnings
import warnings
#Saving the classifier
import pickle
#---------------- Loading data from database ----------------------------------
def load_data(database_filepath):
    '''Function that loads that data from a SQL database and
    returns the feature X and target variables Y'''
    warnings.simplefilter('ignore')
    engine = create_engine('sqlite:///DisasterMessages.db')
    df = pd.read_sql("SELECT * FROM Messages",engine)
    #X = df.iloc[:,1]
    #Y = df.iloc[:,4:]
    X = df['message']
    Y = df.drop(['id','message','original','genre'],axis=1)
    category_names = Y.columns
    #Only for testing
    X = X[:1000]
    Y = Y[:1000]
    return X,Y,category_names
#------------------ Tokenization function to process text data ----------------
def tokenize(text):
    warnings.simplefilter('ignore')
    # Text Normalization(remove any characters that are not letters or numbers)
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    # Tokenize text
    tokens = word_tokenize(text)
    #remove stop words // Note: sometimes removing all the stopwords results in just one
    #single word without a concrete meaning

    #tokens = [w for w in tokens if w not in stopwords.words('english')]

    lemmatizer = WordNetLemmatizer()

    clean_tokens=[]
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

#----------- Building the Machine Learning Pipeline ---------------------------
def build_model():
    '''Function that builds a ML Pipeline with optimization parameters '''
    #Ignore warning messages
    warnings.simplefilter('ignore')

    # build pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(OneVsRestClassifier(RandomForestClassifier())))
     ])
    # parameters for grid search
    parameters = {
        'vect__ngram_range': ((1, 1), (1, 2)),
        'vect__max_df': (0.5, 0.75, 1.0),
        'vect__max_features': (None, 5000, 10000),
        'tfidf__use_idf': (True, False),
        'clf__estimator__estimator__n_estimators': [10, 50, 100],
        'clf__estimator__estimator__min_samples_split': [2,3]

     }

    model = GridSearchCV(pipeline, param_grid = parameters)

    return model

#--------------- Multiclassification report -------------------------------------
def multiclass_report(y_test, y_pred,category_names):
    warnings.simplefilter('ignore')
    #iterating through the columns and calling sklearn's `classification_report` on each.
    for c, column in enumerate(category_names):
        yc_test=y_test[column]
        yc_pred = y_pred[:,c]
        labels = np.unique(yc_pred)
        class_rep = classification_report(yc_test, yc_pred, labels=labels)
        accuracy = (yc_pred == yc_test).mean()
        print("----------------------------------------------------------------------")
        print("Category {}: {}".format(c+1,yc_test.name))
        print("Labels:", labels)
        print("Classification Report:\n", class_rep)
        print("Accuracy: {}\n".format(accuracy))
        print("----------------------------------------------------------------------")

# ------------- Evaluating Model Performance -----------------------------------
def evaluate_model(model, X_test, Y_test, category_names):
    # predict on test data// GridSearchCV calls predict on the estimator with the best found parameters
    y_pred = model.predict(X_test)
    # display results
    print("Results for the Best Model with the Following Parameters\n{}\n".format(model.best_params_))
    multiclass_report(Y_test, y_pred,category_names)


def save_model(model, model_filepath):
    # save the model to disk
    filename = model_filepath
    pickle.dump(best_model, open(filename, 'wb'))


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
