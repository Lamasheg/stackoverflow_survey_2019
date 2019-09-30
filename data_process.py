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
def split_column_content(df,col1,col2=None,delimiter=';'):
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

def split_and_concat(df,col1,col2,delimiter=';'):
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
        columns = row[col1].new_df(delimeter)
        for col in columns:
            new_df.loc[len(new_df)] = [col, row[col2]]
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
    plt.savefig(title)

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


def clean_data(df):
    '''
    INPUT
    df - pandas dataframe 
    
    OUTPUT
    X - A matrix holding all of the variables you want to consider when predicting the response
    y - the corresponding response vector
    
    This function cleans df using the following steps to produce X and y:
    1. Drop all the rows with no salaries
    2. Create X as all the columns that are not the Salary column
    3. Create y as the Salary column
    4. Drop the Salary, Respondent, and the ExpectedSalary columns from X
    5. For each numeric variable in X, fill the column with the mean value of the column.
    6. Create dummy columns for all the categorical variables in X, drop the original columns
    '''
    # Drop rows with missing salary values
    df = df.dropna(subset=['ConvertedComp'], axis=0)
    y = df['ConvertedComp']
    
    #Drop respondent and expected salary columns
    df = df.drop(['Respondent', 'ConvertedComp', 'CompTotal'], axis=1)
    
    # Fill numeric columns with the mean
    num_vars = df.select_dtypes(include=['float', 'int']).columns
    for col in num_vars:
        df[col].fillna((df[col].mean()), inplace=True)
        
    # Dummy the categorical variables
    cat_vars = df.select_dtypes(include=['object']).copy().columns
    for var in  cat_vars:
        # for each cat add dummy var, drop original column
        df = pd.concat([df.drop(var, axis=1), pd.get_dummies(df[var], prefix=var, prefix_sep='_', drop_first=True)], axis=1)
    
    X = df
    return X, y
    
#Use the function to create X and y
# X, y = clean_data(df)        


def find_optimal_lm_mod(X, y, cutoffs, test_size = .30, random_state=42, plot=True):
    '''
    INPUT
    X - pandas dataframe, X matrix
    y - pandas dataframe, response variable
    cutoffs - list of ints, cutoff for number of non-zero values in dummy categorical vars
    test_size - float between 0 and 1, default 0.3, determines the proportion of data as test data
    random_state - int, default 42, controls random state for train_test_split
    plot - boolean, default 0.3, True to plot result

    OUTPUT
    r2_scores_test - list of floats of r2 scores on the test data
    r2_scores_train - list of floats of r2 scores on the train data
    lm_model - model object from sklearn
    X_train, X_test, y_train, y_test - output from sklearn train test split used for optimal model
    '''
    r2_scores_test, r2_scores_train, num_feats, results = [], [], [], dict()
    for cutoff in cutoffs:

        #reduce X matrix
        reduce_X = X.iloc[:, np.where((X.sum() > cutoff) == True)[0]]
        num_feats.append(reduce_X.shape[1])

        #split the data into train and test
        X_train, X_test, y_train, y_test = train_test_split(reduce_X, y, test_size = test_size, random_state=random_state)

        #fit the model and obtain pred response
        lm_model = LinearRegression(normalize=True)
        lm_model.fit(X_train, y_train)
        y_test_preds = lm_model.predict(X_test)
        y_train_preds = lm_model.predict(X_train)

        #append the r2 value from the test set
        r2_scores_test.append(r2_score(y_test, y_test_preds))
        r2_scores_train.append(r2_score(y_train, y_train_preds))
        results[str(cutoff)] = r2_score(y_test, y_test_preds)

    if plot:
        plt.plot(num_feats, r2_scores_test, label="Test", alpha=.5)
        plt.plot(num_feats, r2_scores_train, label="Train", alpha=.5)
        plt.xlabel('Number of Features')
        plt.ylabel('Rsquared')
        plt.title('Rsquared by Number of Features')
        plt.legend(loc=1)
        plt.show()

    best_cutoff = max(results, key=results.get)

    #reduce X matrix
    reduce_X = X.iloc[:, np.where((X.sum() > int(best_cutoff)) == True)[0]]
    num_feats.append(reduce_X.shape[1])

    #split the data into train and test
    X_train, X_test, y_train, y_test = train_test_split(reduce_X, y, test_size = test_size, random_state=random_state)

    #fit the model
    lm_model = LinearRegression(normalize=True)
    lm_model.fit(X_train, y_train)

    return r2_scores_test, r2_scores_train, lm_model, X_train, X_test, y_train, y_test

