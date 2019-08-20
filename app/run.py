#Import libraries
import json
import plotly
import pandas as pd
import re
#NLP libraries
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
#Libraries for backend and visuals
from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine
from collections import Counter

app = Flask(__name__)

def tokenize(text):
    '''
    Funtion to create tokens from input news_messages
    Args: text message
    Return: clean_tokens
    '''
    #Only letters are considered
    text = re.sub(r"[^a-zA-Z]", " ", text.lower())
    tokens = word_tokenize(text)
    #List of stop words not considered
    STOPWORDS = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
    tokens = [w for w in tokens if w not in STOPWORDS]
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# loading the data from the sqlite database
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('Messages', engine)

# load trained model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # extract data needed for visuals
    # Getting genre counts
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # Message counts for different categories
    cat_counts_df = df.iloc[:, 4:].sum().sort_values(ascending=False)
    cat_counts = list(cat_counts_df)
    cat_names = list(cat_counts_df.index)

    # Top keywords in News in percentages
    news_messages = ' '.join(df[df['genre'] == 'news']['message'])
    news_tokens = tokenize(news_messages)
    news_wrd_counter = Counter(news_tokens).most_common()
    news_wrd_cnt = [i[1] for i in news_wrd_counter]
    news_wrd_pct = [i/sum(news_wrd_cnt) *100 for i in news_wrd_cnt]
    news_wrds = [i[0] for i in news_wrd_counter]

    # creating visuals

    graphs = [
    # Histogram of the message genre
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Messages by Genre',
                'yaxis': {
                    'title': "Total Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
         # histogram of messages distributions for each category
        {
            'data': [
                    Bar(
                        x=cat_names,
                        y=cat_counts
                                    )
            ],

            'layout':{
                'title': "Distribution of Message Categories",
                'xaxis': {'tickangle':60
                },
                'yaxis': {
                    'title': "Number of Messages"
                }
            }
        },

        # histogram of news messages for top 50 keywords
        {
            'data': [
                    Bar(
                        x=news_wrds[:50],
                        y=news_wrd_pct[:50]
                                    )
            ],

            'layout':{
                'title': "Top 50 Keywords in Messages on News",
                'xaxis': {'tickangle':60
                },
                'yaxis': {
                    'title': "% Keywords in Messages"
                }
            }
        }
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
