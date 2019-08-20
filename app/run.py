import json
import plotly
import pandas as pd
import re


from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine
from collections import Counter

app = Flask(__name__)

def tokenize(text):
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('Messages', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
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
  
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
    # Histogram of the message genere
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
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
                    'title': "count"    
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
                'title': "Top 50 Keywords in News Messages",
                'xaxis': {'tickangle':60
                },
                'yaxis': {
                    'title': "% Messages on News"    
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