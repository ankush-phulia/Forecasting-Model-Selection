import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
# sklearn models
from sklearn import preprocessing, linear_model, svm, neural_network, ensemble
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import cross_val_score


# display setting
pd.set_option('expand_frame_repr', False)

# Data specific paramerters
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
        help='Which years to analyse for, space separated')
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


def plot(col_name, df, typ='o'):
    '''
    Plot a column of dataframe, given the flag is valid, i.e. not 99
    '''
    # check col_name is not time
    if col_name in ['Year', 'Month', 'Day', 'Hour', 'Minute']:
        return

    index = df.columns.get_loc(col_name)
    col_flags = df.columns[index + 1]
    Y = df[col_name][df[col_flags] < 99]

    plt.plot(Y, typ, label=col_name, markersize=1)
    plt.title('{} vs Time, {}/{}/{} - {}/{}/{}'.format(col_name,
                                                       df['Day'].iloc[0],
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
    # Data Formatting
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


def cleanupDf(df, measure_cols=['GlobalHorizIrr(PSP)', 'DirNormIrr', 'DiffuseHorizIrr'],
                  flag_cols=['GHIFlag', 'DNIFlag', 'DHIFlag']):
    '''
    Remove entries with invalid flags and invalid columns
    '''
    # replace all 99s and -99999.0 with NaN
    for measure, flag in zip(measure_cols, flag_cols):
        df[flag].replace(99, np.nan, inplace=True)
        df.loc[df[flag].isnull(), measure] = np.nan

    # remove invalid columns - all NaNs
    df.dropna(axis=1, how='all', inplace=True)


def averageDf(df, col, 
    possible_cols=['Year', 'Month', 'Day', 'Hour', 'Minute',]):
    '''
    Groups all the observations of df by the cols, and averages over that time span
    '''
    if not(col in possible_cols):
        print 'Cant take average on this column'
        return df, possible_cols

    cols = possible_cols[:possible_cols.index(col) + 1]
    df = df.groupby(cols).mean().reset_index()
    for col in possible_cols:
        if not(col in cols):
            df.drop(col, axis = 1, inplace=True)
    return df, cols


# def createDataSets(df, input_measure_cols=['GlobalHorizIrr(PSP)'],
#                        input_flag_cols=['GHIFlag'],
#                        output_measure_cols=['GlobalHorizIrr(PSP)'],
#                        window=100):
#     '''
#     Create Training and Testing Sets
#     '''


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
    measure_cols = ['GlobalHorizIrr(PSP)', 'DirNormIrr']

    # take daily average
    Data_avg, time_cols = averageDf(Data, 'Day')

    # get correlation between the measure columns
    print 'Correlation in {}'.format(measure_cols)
    print Data[measure_cols].corr()

    # print Data_avg
    plot(measure_cols[1], Data_avg, '-')


if __name__ == '__main__':
    args = vars(getParser().parse_args(sys.argv[1:]))
    Run(args)
