
### Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [Project Components](#components)
4. [Results](#results)
5. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>

The main libraries with corresponding versions to run  the code are the following:   
<ul>
<li>Python 3.6.7 |Anaconda, Inc.|</li>
<li>Scikit-Learn 0.20.2</li>
<li>NumPy 1.15.4</li>
<li>SciPy 1.1.0</li>
<li>Pandas 0.23.4</li>
</ul>


## Project Motivation<a name="motivation"></a>

For this project, I will design an ETL pipeline, machine learning pipeline and use Natural Language Processing (NLP) to
analyze disaster data from [FigureEight](https://www.figure-eight.com/) to build a model for an API that classifies disaster messages.

The project includes a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data.


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
**3. Flask Web App**
Web App that shows data visualizations and allows a person to use the trained model
to classify disaster messages.

## Results<a name="results"></a>

The main findings of the code can be found at the post available [here]().

## Licensing, Authors, Acknowledgements<a name="licensing"></a>

Must give credit to FigureEight for the data and project idea. Author: Gustavo Cedeno following recommendations and requirements from Udacity's Data Science ND Program. 
# NLP_DisasterResponse
