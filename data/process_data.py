import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Load and merge data from CSV files.

    Args:
        messages_filepath (str): Path to the CSV file containing messages data.
        categories_filepath (str): Path to the CSV file containing categories data.

    Returns:
        pandas.DataFrame: Merged DataFrame containing messages and categories data.
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on='id')
    return df


def clean_data(df):
    """
    Clean the merged DataFrame by splitting and formatting the categories.

    Args:
        df (pandas.DataFrame): Merged DataFrame containing messages and categories data.

    Returns:
        pandas.DataFrame: Cleaned DataFrame with formatted categories.
    """
    # Split the categories
    categories = df['categories'].str.split(pat=';', expand=True)
    
    # Fix the category column names
    row = categories.iloc[[1]]
    category_colnames = [category_name.split('-')[0] for category_name in row.values[0]]
    categories.columns = category_colnames
    
    for column in categories:
        categories[column] = categories[column].str[-1]
        categories[column] = categories[column].astype(int)
    
    df = df.drop('categories', axis=1)
    df = pd.concat([df, categories], axis=1)
    df = df.drop_duplicates()
    numbers_columns = df.select_dtypes(include=['number']).columns
    df[numbers_columns] = df[numbers_columns].applymap(lambda x: 1 if pd.to_numeric(x) > 1 else x)
    
    return df


def save_data(df, database_filename):
    """
    Save the cleaned data to a SQLite database.

    Args:
        df (pandas.DataFrame): Cleaned DataFrame with formatted categories.
        database_filename (str): Name of the SQLite database file.
    """
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql(database_filename, engine, index=False)


def main():
    """
    Main function to load, clean, and save data from CSV files to a SQLite database.
    """
    if len(sys.argv) == 4:
        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to the database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second arguments respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
