import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
# sklearn models
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn import preprocessing, linear_model, svm, neural_network, ensemble


# display setting
pd.set_option('expand_frame_repr', False)

# Data specific parameters
numTimeCols = 5


def getParser():
    parser = argparse.ArgumentParser(
        description='Analyse Time Series Data')
    parser.add_argument(
        '--data_dir',
        required=True,
        help='Directory with csv data files')
    parser.add_argument(
        '--years',
        nargs='*',
        default=[],
        help='Which years to analyse for, space separated.\
        Use y1-y2 to specify all years between and including y1 and y2')
    parser.add_argument(
        '--start_month',
        type=int,
        default=0,
        help='Which month to start with')
    parser.add_argument(
        '--num_months',
        type=int,
        default=0,
        help='Number of months to analyse')
    return parser


def setupPlot():
    mpl.rcParams['axes.prop_cycle'] = mpl.cycler('color', ['k'])
    plt.axhline(0, alpha=0.1)
    plt.axhline(0, alpha=0.1)


def closePlot(event):
    if event.key == 'q':
        plt.close(event.canvas.figure)


def plot(col_names, df, typ='o'):
    '''
    Plot a column of dataframe, given the flag is valid, i.e. not 99
    '''
    # check col_name is not time
    for col_name in col_names:
        if col_name in ['Year', 'Month', 'Day', 'Hour', 'Minute']:
            return

        Y = df[col_name]
        plt.plot(Y, typ, label=col_name, markersize=1)

    plt.title('{}/{}/{} - {}/{}/{}'.format(df['Day'].iloc[0],
                                           df['Month'].iloc[0],
                                           df['Year'].iloc[0],
                                           df['Day'].iloc[-1],
                                           df['Month'].iloc[-1],
                                           df['Year'].iloc[-1]))
    plt.legend(loc='upper right')
    plt.show()


def readData(data_dir, years="", start_month=0, num_months=0):
    '''
    Read data from CSV files into DataFrame
    '''
    # Data Formatting - allow specification of year ranges
    expanded_years = []
    for year in years:
        if '-' in year:
            [start_yr, end_yr] = year.split('-')
            expanded_years += [str(i) for i in xrange(int(start_yr), int(end_yr) + 1)]
        else:
            expanded_years.append(year)
    years = expanded_years
    years.sort()

    start_month = max(0, start_month)
    start_month = min(start_month, 12)

    # relevant column names for the site BS
    cols = [
        'Year',
        'Month',
        'Day',
        'Hour',
        'Minute',
        'GlobalHorizIrr(PSP)', 'GHIFlag',
        'DirNormIrr', 'DNIFlag',
        'DiffuseHorizIrr', 'DHIFlag']
    measure_cols = ['GlobalHorizIrr(PSP)', 'DirNormIrr', 'DiffuseHorizIrr']
    flag_cols = ['GHIFlag', 'DNIFlag', 'DHIFlag']

    first_month_found = (len(years) == 0 or start_month == 0)
    dfs = []

    for f in os.listdir(data_dir):
        # check if required number of months is found
        if years and start_month and len(dfs) == num_months:
            break

        if f.endswith('.csv'):
            if start_month == int(f[6:8]):
                first_month_found = True

            if first_month_found and (not(len(years)) or f[2:6] in years):
                # make dataframe out of csv and append it
                dfs.append(
                    pd.read_csv(
                        os.path.join(data_dir, f),
                        usecols=range(len(cols)), header=None, names=cols))

    return dfs, measure_cols, flag_cols


def cleanupDf(df, measure_cols=[
        'GlobalHorizIrr(PSP)',
        'DirNormIrr',
        'DiffuseHorizIrr'],
        flag_cols=['GHIFlag', 'DNIFlag', 'DHIFlag']):
    '''
    Remove entries with invalid flags and invalid columns
    '''
    # remove negative measures
    numbers = df._get_numeric_data()
    numbers[numbers < 0] = np.nan

    # replace all -ves, 99 flags and -99999.0 with NaN
    for measure, flag in zip(measure_cols, flag_cols):
        df[flag].replace(99, np.nan, inplace=True)
        df.loc[df[flag].isnull(), measure] = np.nan

    # remove invalid rows/columns - all NaNs
    df.dropna(axis=0, how='all', subset=measure_cols, inplace=True)
    df.dropna(axis=1, how='all', inplace=True)


def aggregateDf(df, col, operation='avg',
                possible_cols=['Year', 'Month', 'Day', 'Hour', 'Minute']):
    '''
    Groups all the observations of df by the cols, and aggregates over that time span
    '''
    if not(col in possible_cols):
        print 'Cant take aggregate on this column'
        return df, possible_cols

    cols = possible_cols[:possible_cols.index(col) + 1]
    if operation == 'avg':
        df = df.groupby(cols).mean().reset_index()
    elif operation == 'sum':
        df = df.groupby(cols).sum().reset_index()

    # remove columns of smaller time values
    for col in possible_cols:
        if not(col in cols):
            df.drop(col, axis=1, inplace=True)

    return df, cols


def createDataSets(df, input_measure_cols=['GlobalHorizIrr(PSP)'],
                   input_flag_cols=['GHIFlag'],
                   output_measure_cols=['GlobalHorizIrr(PSP)'],
                   window=7, split_factor=0.1):
    '''
    Create Data set and split into train/test
    '''
    Input, Output = [], []
    for index, row in df.iterrows():
        final = index + window
        if final < len(df):
            Input.append(df[input_measure_cols].iloc[index:final])
            Output.append(df[output_measure_cols].iloc[final])
        else:
            break

    # create training and testing sets
    t_in, test_in, t_out, test_out = train_test_split(
        Input, Output, test_size=split_factor)
    train_in, val_in, train_out, val_out = train_test_split(
        t_in, t_out, test_size=split_factor)

    # print information
    print 'Input  : Past {} days\' {}\nOutput : Current {}'.format(
        window, input_measure_cols, output_measure_cols)
    print 'Constructed {} samples from {} observations'.format(
        len(Input), len(df))
    print 'Training samples   : {}'.format(len(train_in))
    print 'Validation samples : {}'.format(len(val_in))
    print 'Testing samples    : {}'.format(len(test_in))

    return train_in, train_out, val_in, val_out, test_in, test_out


# def crossValidate(df, model, n_fold,
# 	cols_to_use=['GlobalHorizIrr(PSP)', 'GHIFlag',
# 				 'DirNormIrr', 'DNIFlag',
# 				 'DiffuseHorizIrr', 'DHIFlag'],
#  	cols_to_pred=['GlobalHorizIrr(PSP)','GHIFlag',
#  				  'DirNormIrr', 'DNIFlag',
#  				  'DiffuseHorizIrr', 'DHIFlag']):
# 	if model == 'Linear'


def Run(args):
    cid = plt.gcf().canvas.mpl_connect('key_press_event', closePlot)

    # read data to make list of dataframes
    dfs, measure_cols, flag_cols = readData(
        args['data_dir'], args['years'], args['start_month'], args['num_months'])

    # clean-up for each one
    for df in dfs:
        cleanupDf(df, measure_cols, flag_cols)

    # put them all together
    Data = pd.concat(dfs).reset_index(drop=True)
    measure_cols = ['GlobalHorizIrr(PSP)', 'DirNormIrr', 'DiffuseHorizIrr']

    # take sum on a given time scale
    Data_sum, time_cols = aggregateDf(Data, 'Day', 'sum')

    # get correlation between the measure columns
    print 'Correlation in {}'.format(measure_cols)
    print Data_sum[measure_cols].corr()

    # plot Data
    plot(measure_cols, Data_sum, '-')

    # create data set
    train_in, train_out, val_in, val_out, test_in, test_out = createDataSets(
        Data_sum)


if __name__ == '__main__':
    args = vars(getParser().parse_args(sys.argv[1:]))
    Run(args)
