import sys
import pandas as pd


def load_data(messages_filepath, categories_filepath):
    '''
    Input:
    messages_filepath: Route for the disaster_messages.csv file
    categories_filepath: Route for the disaster_categories.csv file
    
    Output:
    df: Merged dataframe
    
    The function merges messages with categories and returns the merged df
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages,categories,on='id')
    return(df)

def clean_data(df):
    '''
    Input:
    df: Raw dataframe to be cleaned
    
    Output:
    df: Cleaned df
    
    This function cleans the df returning a dataframe with separated categories, the boolean value and the duplicates dropped
    '''
    categories = df["categories"].str.split(";", expand=True)
    # select the first row of the categories dataframe
    row = categories.iloc[0]
    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = [x.split('-')[0] for x in row.values]
    
    # rename the columns of `categories`
    categories.columns = category_colnames
    categories = categories.applymap(lambda s: 1 if int(s[-1]) >= 1 else 0)
    
    # drop the original categories column from `df`
    df = df.drop(columns=["categories"])
    df = pd.concat([df, categories], axis=1)
    df.drop_duplicates(inplace=True)
    return(df)
    
def save_data(df, database_filename):
    '''
    Input:
    df: Dataframe
    database_filename: Filepath
    
    Output: None
    
    The function saves the cleaned dataset into an database engine
    '''
    from sqlalchemy import create_engine

    engine = create_engine('sqlite:///Messages.db')
    df.to_sql('Messages', engine, index=False, if_exists='replace')


def main():
    '''
    Input: None
    Output: None
    
    This function orchestates the data cleaning and saving process
    '''
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
