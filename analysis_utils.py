import numpy as np
import pandas as pd
import string
import matplotlib.pyplot as plt
import math
import seaborn as sns
from scipy.stats import norm


def missing_heat_map(DataFrame):
    
    fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (18, 6))
    sns.heatmap(DataFrame.isnull(), yticklabels=False, ax = ax, cbar=False,\
                cmap='viridis')
    ax.set_title('dataset')
    plt.show()
        
    # Calculate the missing values to get a percentage 

    for i in DataFrame:
        print(i,': %',int((DataFrame[i].isnull().sum()/len(DataFrame[i]))*100),\
            'With {} missing values'.format((DataFrame[i].isnull().sum())))


def no_outlier(Data_column,data_set):

    """
    This function will give a brief description of the distribution of data with and without outliers
    Arguments:
    Data_column: takes a string of the name of the column
    data_set: takes the data frame without parentheses

    Returns:
    four plots and a brief description of the data distribution
    """

    X = data_set[Data_column] #set the dataframe
    no_outlier = [] 
    confidence = []
    
    q1 = float(X.describe()['25%']) #get the q1 from the describe function 
    q3 = float(X.describe()['75%']) #get the q3 from the describe function
    iqr = (q3 - q1)*1.5 #get the iqr
    std = float(X.describe()['std']) #get the standered deviation
    mean = float(X.mean()) #get mean
    lower_limit = mean-(1.645*(std/np.sqrt(len(X)))) # calculate the lower limit for 90% confidence
    higher_limit = mean+(1.645*(std/np.sqrt(len(X)))) # calculate the higher limit for 90% confidence
    
    for total in X: #iterate over the data
        if lower_limit < total < higher_limit:
            confidence.append(total) #if the value is in the 90% confidence append to confidence 
        
        if (q1 - iqr) < (total) < (q3 + iqr):
            no_outlier.append(total) #if the value is between the outliers append it to list
        else:
            pass
    #print result
    print('Tukeys method number of outliers is {}'.format((len(X)-len(sorted(no_outlier)))))
    print('90% confidence interval has {} values between {} and {}'.format(len(sorted(confidence)),\
        round(lower_limit),round(higher_limit)))
    #plot 
    fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(12, 12))
    sns.distplot(X, ax=ax[0,0])
    sns.distplot(no_outlier,color='red', ax=ax[0,1])
    sns.boxplot(X,notch=True,orient='v',ax=ax[1,0])
    sns.boxplot(no_outlier,notch=True,orient='v',color='red',ax=ax[1,1])
    
    fig.suptitle('{}'.format(Data_column), fontsize=24)
    ax[0,0].set_title('Distribution of {}'.format(Data_column), fontsize=12)
    ax[0,1].set_title('Distribution of {} after removing outliers'.format(Data_column), fontsize=10)
    ax[1,0].set_title('Boxplot of {}'.format(Data_column), fontsize=10)
    ax[1,1].set_title('Boxplot of {} after removing outliers'.format(Data_column), fontsize=10)


def get_wordcount(column,df):

    """
    This function will give you the word count each row in the column
    Arguments:
    column: takes a string of the name of the column
    df: takes the data frame without parentheses

    Returns:
    pandas.core.series containing the wordcounts
    """

    return df[str(column)].apply(lambda x : len(x.split(' ')) if type(x) == str else 0)


def get_number(str):

    """
    This function will return only the numbers from a string
    Arguments:
    str: takes a string
 
    Returns:
    a float containing the digits in the string
    """

    return float(re.sub("[^0-9]", "", str))


def plot_corr(df):

    """
    This function will plot the correlation
    Arguments:
    df: takes a data frame 
 
    Returns:
    a plot of the correlations in the data frame
    """

    plt.figure(figsize=(15,5))
    corr=df.corr()
    sns.set(font_scale=2.5)
    sns.heatmap(corr,annot=True, vmin=0, vmax=1, cmap = 'gist_heat_r')


def plot_line_correlation(dependent,target,dataframe,color='red'):

    """
    This function will plot a scatter plot of a dependent variable to a target 
    Arguments:
    dependent: take a list of column names for the dependent variable (Maximum = 8)
    target: takes a string with the target column name
    data frame:Takes a data frame name without parentheses
    color: optional- choose the color of plots

 
    Returns:
    a plot for every dependent variable in the list
    """

    if len(dependent) == 1:
        ncols = 1 #specify the number of columns
        nrows = 1 #specify the number of rows 
        fig, ax = plt.subplots(ncols=ncols, nrows=ncols, figsize=(12, 12)) #Intoduce a figure that includes the number of graphs
        sns.regplot(dependent[0], target, data=dataframe, fit_reg=True,color=color,scatter_kws={'s':10})
        plt.show()
    elif len(dependent) == 2:
        ncols = 2 #specify the number of columns
        nrows = 2 #specify the number of rows
        fig, ax = plt.subplots(ncols=ncols, nrows=nrows, figsize=(12, 12)) #Intoduce a figure that includes the number of graphs
        sns.regplot(dependent[0], target, data=dataframe, fit_reg=True,color=color,scatter_kws={'s':10}, ax=ax[1,0]) 
        sns.regplot(dependent[1], target, data=dataframe, fit_reg=True,color=color,scatter_kws={'s':10}, ax=ax[0,1])
        plt.show()
    
    elif len(dependent) == 3:
        ncols = 2 #specify the number of columns
        nrows = 2 #specify the number of rows
        fig, ax = plt.subplots(ncols=ncols, nrows=nrows, figsize=(12, 12)) #Intoduce a figure that includes the number of graphs
        sns.regplot(dependent[0], target, data=dataframe, fit_reg=True,color=color,scatter_kws={'s':10}, ax=ax[0,0]) 
        sns.regplot(dependent[1], target, data=dataframe, fit_reg=True,color=color,scatter_kws={'s':10}, ax=ax[0,1])
        sns.regplot(dependent[2], target, data=dataframe, fit_reg=True,color=color,scatter_kws={'s':10}, ax=ax[1,0])
        plt.show()
        
    elif len(dependent) == 4:
        ncols = 2 #specify the number of columns
        nrows = 2 #specify the number of rows
        fig, ax = plt.subplots(ncols=ncols, nrows=nrows, figsize=(12, 12)) #Intoduce a figure that includes the number of graphs
        sns.regplot(dependent[0], target, data=dataframe, fit_reg=True,color=color,scatter_kws={'s':10}, ax=ax[0,0]) 
        sns.regplot(dependent[1], target, data=dataframe, fit_reg=True,color=color,scatter_kws={'s':10}, ax=ax[0,1])
        sns.regplot(dependent[2], target, data=dataframe, fit_reg=True,color=color,scatter_kws={'s':10}, ax=ax[1,0])
        sns.regplot(dependent[3], target, data=dataframe, fit_reg=True,color=color,scatter_kws={'s':10}, ax=ax[1,1])
        plt.show()
        
    elif len(dependent) == 5:
        ncols = 2 #specify the number of columns
        nrows = 3 #specify the number of rows
        fig, ax = plt.subplots(ncols=ncols, nrows=nrows, figsize=(12, 12)) #Intoduce a figure that includes the number of graphs
        sns.regplot(dependent[0], target, data=dataframe, fit_reg=True,color=color,scatter_kws={'s':10}, ax=ax[0,0]) 
        sns.regplot(dependent[1], target, data=dataframe, fit_reg=True,color=color,scatter_kws={'s':10}, ax=ax[0,1])
        sns.regplot(dependent[2], target, data=dataframe, fit_reg=True,color=color,scatter_kws={'s':10}, ax=ax[1,0])
        sns.regplot(dependent[3], target, data=dataframe, fit_reg=True,color=color,scatter_kws={'s':10}, ax=ax[1,1])
        sns.regplot(dependent[4], target, data=dataframe, fit_reg=True,color=color,scatter_kws={'s':10}, ax=ax[2,0])
        plt.show()
        
    elif len(dependent) == 6:
        ncols = 2 #specify the number of columns
        nrows = 3 #specify the number of rows
        fig, ax = plt.subplots(ncols=ncols, nrows=nrows, figsize=(12, 12)) #Intoduce a figure that includes the number of graphs
        sns.regplot(dependent[0], target, data=dataframe, fit_reg=True,color=color,scatter_kws={'s':10}, ax=ax[0,0]) 
        sns.regplot(dependent[1], target, data=dataframe, fit_reg=True,color=color,scatter_kws={'s':10}, ax=ax[0,1])
        sns.regplot(dependent[2], target, data=dataframe, fit_reg=True,color=color,scatter_kws={'s':10}, ax=ax[1,0])
        sns.regplot(dependent[3], target, data=dataframe, fit_reg=True,color=color,scatter_kws={'s':10}, ax=ax[1,1])
        sns.regplot(dependent[4], target, data=dataframe, fit_reg=True,color=color,scatter_kws={'s':10}, ax=ax[2,0])
        sns.regplot(dependent[5], target, data=dataframe, fit_reg=True,color=color,scatter_kws={'s':10}, ax=ax[2,1])
        plt.show()
        
    elif len(dependent) == 7:
        ncols = 3 #specify the number of columns
        nrows = 3 #specify the number of rows
        fig, ax = plt.subplots(ncols=ncols, nrows=nrows, figsize=(12, 12)) #Intoduce a figure that includes the number of graphs
        sns.regplot(dependent[0], target, data=dataframe, fit_reg=True,color=color,scatter_kws={'s':10}, ax=ax[0,0]) 
        sns.regplot(dependent[1], target, data=dataframe, fit_reg=True,color=color,scatter_kws={'s':10}, ax=ax[0,1])
        sns.regplot(dependent[2], target, data=dataframe, fit_reg=True,color=color,scatter_kws={'s':10}, ax=ax[1,0])
        sns.regplot(dependent[3], target, data=dataframe, fit_reg=True,color=color,scatter_kws={'s':10}, ax=ax[1,1])
        sns.regplot(dependent[4], target, data=dataframe, fit_reg=True,color=color,scatter_kws={'s':10}, ax=ax[2,0])
        sns.regplot(dependent[5], target, data=dataframe, fit_reg=True,color=color,scatter_kws={'s':10}, ax=ax[2,1])
        sns.regplot(dependent[6], target, data=dataframe, fit_reg=True,color=color,scatter_kws={'s':10}, ax=ax[1,2])
        plt.show()

    elif len(dependent) == 8:
        ncols = 3 #specify the number of columns
        nrows = 3 #specify the number of rows
        fig, ax = plt.subplots(ncols=ncols, nrows=nrows, figsize=(12, 12)) #Intoduce a figure that includes the number of graphs
        sns.regplot(dependent[0], target, data=dataframe, fit_reg=True,color=color,scatter_kws={'s':10}, ax=ax[0,0]) 
        sns.regplot(dependent[1], target, data=dataframe, fit_reg=True,color=color,scatter_kws={'s':10}, ax=ax[0,1])
        sns.regplot(dependent[2], target, data=dataframe, fit_reg=True,color=color,scatter_kws={'s':10}, ax=ax[1,0])
        sns.regplot(dependent[3], target, data=dataframe, fit_reg=True,color=color,scatter_kws={'s':10}, ax=ax[1,1])
        sns.regplot(dependent[4], target, data=dataframe, fit_reg=True,color=color,scatter_kws={'s':10}, ax=ax[2,0])
        sns.regplot(dependent[5], target, data=dataframe, fit_reg=True,color=color,scatter_kws={'s':10}, ax=ax[2,1])
        sns.regplot(dependent[6], target, data=dataframe, fit_reg=True,color=color,scatter_kws={'s':10}, ax=ax[1,2])
        sns.regplot(dependent[7], target, data=dataframe, fit_reg=True,color=color,scatter_kws={'s':10}, ax=ax[2,2])
        plt.show()


def replace_mean(column,df):
    
    """
    The function will return the column after filling the missing values with the mean
    Arguments:
    column: column name in string format
    df: Dataframe name without parentheses
    
    Returns:
    The inserted column after filling the missing values with the mean
    
    """

    return df[str(column)].fillna(df[str(column)].mean(),inplace = True)


def change_tobool(column,df):

    """
    This function changes the t,f to integer 1,0
    Arguments:
    column: column name in string format
    df: Dataframe name without parentheses
 
    Returns:
    The inserted column after replacing t,f values to 1,0
    
    """

    df[str(column)] = df[str(column)].apply(lambda x : 1 if x == 't' else 0)


def magnify_corr(dataframe):

    """
    This function will plot the correlation using an interactive plot
    Arguments:
    dataframe: takes a dataframe 
 
    Returns:
    a plot of the correlations in the data frame
    """
    
    cmap=sns.diverging_palette(5, 250, as_cmap=True)
    corr = dataframe.corr()
    
    a = [dict(selector="th",props=[("font-size","7pt")]),
     dict(selector="td",props=[('padding',"0em 0em")]),
     dict(selector="th:hover",props=[("font-size","12pt")]),
     dict(selector="tr:hover td:hover",props=[('max-width','200px'),('font-size','12pt')])]
    
    
    return corr.style.background_gradient(cmap, axis=1)\
        .set_properties(**{'max-width': '80px', 'font-size': '10pt'})\
        .set_caption("Hover to magify")\
        .set_precision(2)\
        .set_table_styles(a)


def top_corr_features(target,number,dataframe):
    """
    This function will give you the required number of highest correlated features to a target 
    Arguments:
    target: column name in string format of the targeted column
    number: The number of highest correlated features required
    data frame: the data frame name without parenthesis 

    returns: A list of the highest correlated features with the target (will also print the list)
    """
    best_feature_corr=dataframe.corr()[str(target)].sort_values(ascending=False)\
        .index[0:int(number)].tolist()
    print('list of {} best positive features based on pairwise correlation:\n'\
        .format(number),best_feature_corr)
    return best_feature_corr

def dist_plot(x, la, co):

    plt.figure(figsize=(14,6))
    sns.distplot(x, label=la, fit=norm, color=co)
    plt.xticks(rotation=45)
    plt.legend(fontsize='14')
    
def bar_plot(x, y, title):
    
    # Set up barplot 
    plt.figure(figsize=(14,8))
    g = sns.barplot(x, y)    
    ax=g

    # Enable bar values
    # create a list to collect the plt.patches data
    totals = []

    # Label the graph
    plt.title(title, fontsize = 20)
    plt.xticks(fontsize = 10)

    # find the values and append to list
    for p in ax.patches:
        totals.append(p.get_width())

    # set individual bar lables using above list
    total = sum(totals)

    # set individual bar lables using above list
    for p in ax.patches:
        # get_width pulls left or right; get_y pushes up or down
        ax.text(p.get_width()+.3, p.get_y()+.38, \
                int(p.get_width()), fontsize=10)
