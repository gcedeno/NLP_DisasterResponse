
### Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [Project Components](#components)
4. [Instructions](#instructions)
5. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>

The main libraries needed to run  the code are the following:   
<ul>
<li>Python 3.6.7 |Anaconda, Inc.|</li>
<li>Scikit-Learn for Machine Learning algorithms</li>
<li>NumPy for numerical vectorize calculations</li>
<li>Pandas for data manipulation</li>
<li>NLTK for Natural Language Processing</li>
<li>sqlalchemy for interaction within databases</li>
</ul>


## Project Motivation<a name="motivation"></a>

For this project, I designed an ETL pipeline, machine learning pipeline that uses Natural
Language Processing (NLP) to analyze disaster data from [FigureEight](https://www.figure-eight.com/)
to build a model for an API that classifies disaster messages.

The project includes a web app where an emergency worker can input a new message and get
classification results in several categories. The web app will also display visualizations of the data.


## Project Components <a name="components"></a>
There are three main components for this project:

**1. ETL Pipeline**

The Python script, `process_data.py`, contains a data cleaning pipeline that:
<ul>
<li>Loads the messages and categories datasets</li>
<li>Merges the two datasets</li>
<li>Cleans the data</li>
<li>Stores it in a SQLite database</li>
</ul>

**2. ML Pipeline**

In the Python script, `train_classifier.py`, there is a machine learning pipeline that:
<ul>
<li>Loads data from the SQLite database</li>
<li>Splits the dataset into training and test sets</li>
<li>Builds a text processing and machine learning pipeline</li>
<li>Trains and tunes a model using GridSearchCV</li>
<li>Outputs results on the test set</li>
<li>Exports the final model as a pickle file</li>
</ul>

**3. Flask Web App**

Web App that shows data visualizations and allows a person to use the trained model
to classify disaster messages.

## Instructions<a name="instructions"></a>

1. Run the following commands in the project's root directory to set up the database and model.

- To run ETL pipeline that cleans data and stores in database

    `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
- To run ML pipeline that trains classifier and saves

    `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run the web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/ (Go to your web app link)

## Licensing, Authors, Acknowledgements<a name="licensing"></a>

Must give credit to FigureEight for the data and project idea. Author: Gustavo Cedeno following
recommendations and requirements from Udacity's Data Science ND Program.
# NLP_DisasterResponse
