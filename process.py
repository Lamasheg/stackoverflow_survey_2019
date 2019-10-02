import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

 
 
 # Function to show the ratio and count of missing data
def missing_ratio(df):
    '''
    INPUT: 
        - df : data frame to check
    OUTPUT: 
        - new_df : new dataframe with counts and ratio of missing values in each category
    '''            

    count = df.isnull().sum().sort_values(ascending = False)
    ratio = (df.isnull().sum()/df.isnull().count()*100).sort_values(ascending = False)
    new_df = pd.concat([count, ratio], axis=1, keys=['Count', 'Ratio'])
    return new_df
# Function to extract a certain column's question description

def get_description(column_name, schema):
    '''
    INPUT - schema - pandas dataframe with the schema of the developers survey
            column_name - string - the name of the column you would like to know about
    OUTPUT - 
            desc - string - the description of the column
    '''
    desc = list(schema[schema['Column'] == column_name]['QuestionText'])[0]
    return desc

# Function to separate strings in a column content
def split_column_content(df, col1, col2=None, delimiter=';'):
    '''
    INPUT:
        - df : a dataframe of inerest
        - col - string : a column for splitting
        - delimiter - string : a character that seperates the strings in a column content
    OUTPUT:
        - new_df : a new dataframe 
    '''
    new_df = pd.DataFrame(df[col1].dropna().str.split(delimiter).tolist()).stack()
    new_df.reset_index(drop=True)
    return new_df

# Function to split strings in a column content and concat with salary column
def split_and_concat(df, col1, col2, delimiter=';'):
    '''
    INPUT:
        - df : a dataframe of inerest
        - col1 - string : a column for splitting
        - col2 = string : a column you want to concat to col1 after col1 has been split
        - delimiter - string : a character that seperates the strings in a column content
    OUTPUT:
        - new_df : a new dataframe 
    '''
    new_df = pd.DataFrame(columns = [col1, col2])
    for index, row in df.iterrows():
        columns = row[col1].split(delimiter)
        for col in columns:
            new_df.loc[len(new_df)] = [col, row[col2]]
    return new_df

# Function to group a column and apply aggregation function to another coloumn
def group_and_agg(df, col1, col2='ConvertedComp',agg_type='median' ):
    '''
    INPUT:
        - df : a dataframe of inerest
        - col1 - string : a column you want to group
        - col2 - string : a column you want to apply aggregation
        - agg_type - string : how you want col2 to be calculated
    OUTPUT:
        - new_df : a new dataframe 
    '''
    new_df = df.groupby(col1).agg({col2:agg_type}).reset_index().sort_values(col2,ascending=False)
    return new_df

def count_and_plot(s,title):
    '''
    INPUT:
        - s : a pd series (a column sliced from a dataframe) to perform value_count
        - col - string : the name of the column you would like to know about
        - title - string : the title of the chart
    OUTPUT:
        - vc - number : the count of each attribute in chosen column
    '''
    ratio = s.value_counts()/s.shape[0]
    df = pd.DataFrame(pd.Series(ratio)).reset_index()
    df.columns = ['type','ratio']
    print(df.head(10))
    ratio[:10].plot(kind='barh')
    plt.title(title)
    plt.grid(axis='x',linestyle='--')
    plt.xlabel('Ratio')
    # plt.savefig(title)

def plot_value_counts(df, col):
    '''
    INPUT:
        - df : dataframe 
        - col - string : the name of the column you would like to know about
    OUTPUT:
        - vc - number : the count of each attribute in chosen column
    '''
    value_count = df[col].value_counts()
    print(value_count[:10]/df.shape[0])
    (value_count[:10]/df.shape[0]).plot(kind='barh')
    plt.title(col)
    plt.grid(axis='x',linestyle='--')
    plt.xlabel('Ratio')


def total_count(df, col1, col2, look_for):
    '''
    INPUT:
        - df : the pandas dataframe you want to search
        - col1 - string : the column name you want to look through
        - col2 - string : the column you want to count values from
        - look_for : a list of strings you want to search for in each row of df[col]
    
    OUTPUT:
        new_df - a dataframe of each look_for with the count of how often it shows up
    '''		
    new_df = defaultdict(int)
    #loop through list of ed types
    for val in look_for:
        #loop through rows
        for idx in range(df.shape[0]):
            #if the ed type is in the row add 1
            if val in df[col1][idx]:
                new_df[val] += int(df[col2][idx])
    new_df = pd.DataFrame(pd.Series(new_df)).reset_index()
    new_df.columns = [col1, col2]
    new_df.sort_values('count', ascending=False, inplace=True)
    return new_df