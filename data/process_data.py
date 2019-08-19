import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
'''  - To run ETL pipeline that cleans data and stores in database from the terminal
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db` '''

def load_data(messages_filepath, categories_filepath):
    #Loading data and creating DataFrames
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    # merge datasets
    df = messages.merge(categories,how = 'outer',on='id')
    # create a dataframe of the 36 individual category columns
    categories = categories['categories'].str.split(pat=';',expand=True)
    #Using first row of categories dataframe to create column names for the categories data
    #Rename columns of categories with new column names
    row = categories.iloc[0]
    category_colnames = list(row.str.split('-').str.get(0))
    # rename the columns of `categories`
    categories.columns = category_colnames
    #Converting category values to numbers 0 or 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x.split('-')[1])

        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])

    #Converting 2-values to 1
    categories.loc[categories['related'] == 2, 'related'] = 1
    # drop the original categories column from `df`
    df.drop('categories',axis=1,inplace=True)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories],axis=1,sort=False)

    return df


def clean_data(df):
    # drop duplicates
    df.drop_duplicates(inplace=True)
    #Replacing NaNs with 2
    df.fillna(2,inplace=True)

    return df


def save_data(df, database_filename):
    engine = create_engine('sqlite:///'+ database_filename)
    df.to_sql('Messages', engine, index=False, if_exists='replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
