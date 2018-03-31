import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
# statistical models
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa import arima_model
# sklearn models
from sklearn.model_selection import train_test_split, cross_val_score
# from sklearn.pipeline import make_pipeline, Pipeline
# from sklearn import preprocessing, linear_model, svm, neural_network, ensemble


# display setting
pd.set_option('expand_frame_repr', False)

# Data specific parameters
numTimeCols = 5


def getParser():
    parser = argparse.ArgumentParser(
        description='Analyse Time Series Data')
    parser.add_argument(
        '--data_dir',
        help='Directory with csv data files')
    parser.add_argument(
        '--data',
        default="",
        help='One csv file with all the data')
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


def plot(col_names, df, typ='o', combined=True):
    '''
    Plot a column of dataframe, given the flag is valid, i.e. not 99
    '''
    # check col_name is not time
    for col_name in col_names:
        if col_name not in df.columns:
            continue
        Y = df[col_name]
        plt.plot(Y, typ, label=col_name, markersize=1)

        plt.title('{} - {}'.format(
            df.index[0].strftime('%d/%m/%Y'), df.index[-1].strftime('%d/%m/%Y')))
        plt.legend(loc='upper left')
        if not combined:
            plt.show()

    if combined:
        plt.show()


def expandRange(data):
    '''
    Expects a string list of ranges/values - which are ints actually
    Expands the strings which have '-' in them, to all values between them
    '''
    expanded = []
    for data_item in data:
        if '-' in data_item:
            [start, end] = data_item.split('-')
            expanded += [str(i) for i in xrange(int(start), int(end) + 1)]
        else:
            expanded.append(data_item)
    return expanded


def readData(data_dir, data="", years="", start_month=0, num_months=0):
    '''
    Read data from CSV files into DataFrame
    '''
    # relevant column names for the site BS
    cols = ['Timestamp',
            'GlobalHorizIrr(PSP)', 'GHIFlag',
            'DirNormIrr', 'DNIFlag',
            'DiffuseHorizIrr', 'DHIFlag']
    measure_cols = [cols[i] for i in xrange(1, len(cols), 2)]
    flag_cols = [cols[i] for i in xrange(2, len(cols), 2)]

    # data is already in a file processed before
    if len(data):
        df = pd.read_csv(data, index_col=0)
        df.index = pd.to_datetime(df.index)
        return [df], measure_cols, flag_cols

    # Data Formatting - allow specification of year ranges
    years = sorted(expandRange(years))
    start_month = max(0, start_month)
    start_month = min(start_month, 12)

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
                df = pd.read_csv(
                        os.path.join(data_dir, f),
                        usecols=range(numTimeCols + len(measure_cols) * 2),
                        header=None, parse_dates=[[0, 1, 2, 3, 4]])
                # rename columns and set index to time
                df.columns = cols
                df.set_index('Timestamp', inplace=True)
                df.index = pd.to_datetime(df.index, format='%Y %m %d %H %M')
                dfs.append(df)

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

    # remove invalid rows - all NaNs
    df.dropna(axis=0, how='all', subset=measure_cols, inplace=True)
    # df.dropna(axis=1, how='all', inplace=True)


def aggregateDf(df, col, operation='avg',
                possible_cols=['Y', 'M', 'D', 'h', 'm']):
    '''
    Groups all the observations of df by the cols, and aggregates over that time span
    '''
    if not(col in possible_cols):
        print 'Cant take aggregate on this column'
        return df, possible_cols

    if operation == 'avg':
        df = df.resample(col).mean()
    elif operation == 'sum':
        df = df.resample(col).sum()

    df = df.replace(0, np.nan).dropna()
    return df


def createDataSets(df, input_measure_cols=['GlobalHorizIrr(PSP)'],
                   input_flag_cols=['GHIFlag'],
                   output_measure_cols=['GlobalHorizIrr(PSP)'],
                   window=7, split_factor=0.1, split=True):
    '''
    Create Data set and split into train/test
    '''
    Input, Output = [], []
    current_window = df[input_measure_cols].iloc[0:window]
    for index in xrange(window, df.shape[0] - window - 1):
        Input.append(current_window)

        # get new obs - this is output
        new_obs = df[output_measure_cols].iloc[index]
        Output.append(new_obs)

        # modify sliding window
        current_window = current_window.append(new_obs).drop(current_window.index[0])

    if not split:
        return Input, Output

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


def checkStationarity(df, measure_col, plot=False, silent=False):
    '''
    Using rolling stats and Dickey-Fuller to check stationarity
    '''
    ts = df[measure_col]

    # Determing rolling statistics
    rolmean = pd.rolling_mean(ts, window=12)
    rolstd = pd.rolling_std(ts, window=12)

    # Plot rolling statistics:
    if plot:
        orig = plt.plot(ts, label='Original')
        mean = plt.plot(rolmean, label='Rolling Mean')
        std = plt.plot(rolstd, label = 'Rolling Std')
        plt.legend(loc='best')
        plt.title('Rolling Mean & Standard Deviation for {}'.format(measure_col))
        plt.show()
    
    # Perform Dickey-Fuller test:
    dftest = adfuller(ts, autolag='AIC')
    dfoutput = pd.Series(
        dftest[0:4], index=
        ['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    
    # check test statistic vs critical values - get confidence
    confidence = 0
    for key, value in dftest[4].iteritems():
        dfoutput['Critical Value ({})'.format(key)] = value
        if dfoutput['Test Statistic'] < value:
            confidence = max(confidence, 100 - int(key[:-1]))
    
    # print the results
    if not silent:
        print 'Dicky Fuller for {} : '.format(measure_col)
        print dfoutput, '\n'

    return confidence


def plotACF_PACF(df, measure_col):
    '''
    Plot the autocorrelation and partial autocorrelation plots
    '''
    ts = df[measure_col]
    lag_acf = acf(ts, nlags=200)
    lag_pacf = pacf(ts, method='ols', nlags=100)

    # plot acf
    plt.plot(lag_acf)
    plt.axhline(y=0, linestyle='--', color='gray')
    plt.axhline(y= -1.96 / np.sqrt(len(ts)),linestyle='--',color='gray')
    plt.axhline(y= 1.96 / np.sqrt(len(ts)),linestyle='--',color='gray')
    plt.title('Autocorrelation Function for {}'.format(measure_col))
    plt.show()

    # plot pacf
    plt.plot(lag_acf)
    plt.axhline(y=0, linestyle='--', color='gray')
    plt.axhline(y= -1.96 / np.sqrt(len(ts)),linestyle='--',color='gray')
    plt.axhline(y= 1.96 / np.sqrt(len(ts)),linestyle='--',color='gray')
    plt.title('Partial Autocorrelation Function for {}'.format(measure_col))
    plt.show()

    # dislay autocorrelation plots
    pd.tools.plotting.autocorrelation_plot(
        df[measure_col], label=measure_col)
    plt.title('Autocorrelation for {}'.format(measure_col))
    plt.legend(loc='upper right')
    plt.show()


def statModel(df, measure_cols=['GlobalHorizIrr(PSP)']):
    '''
    Model Time series data with statistical models
    '''
    for col in measure_cols:
        # checkStationarity(df, col)
        # plotACF_PACF(df, col)
        ts = df[col]
        p, q, d = 80, 80, 0
        model = arima_model.ARIMA(ts, order=(p, d, q)).fit()
        plt.plot(ts)
        plt.plot(model.fittedvalues, color='red')
        plt.title('RSS : {}'.format(sum((model.fittedvalues - ts)**2)))

    # p, q, r = 20, 1, 0
    # for col in measure_cols:
    #     # fit a model - rolling window style
    #     predictions = []
    #     for i in xrange(len(Input)):
    #         input_sample = Input[i]
    #         output_sample = Output[i]
    #         model = arima_model.ARIMA(input_sample, order=(p, q, r)).fit(disp=0)
    #         residuals = pd.DataFrame(model.resid)

    #         # predict using the model
    #         model_out = model.forecast()
    #         predictions.append(model_out[0])
    
    #     # get the errors and plot
    #     diqff = Output - predictions
    #     plt.plot(Output)
    #     plt.plot(predictions)
    #     plt.show()


def Run(args):
    cid = plt.gcf().canvas.mpl_connect('key_press_event', closePlot)

    # read data to make list of dataframes
    dfs, measure_cols, flag_cols = readData(
        args['data_dir'], args['data'], args['years'], args['start_month'], args['num_months'])

    # clean-up for each one
    for df in dfs:
        cleanupDf(df, measure_cols, flag_cols)

    # put them all together
    Data = pd.concat(dfs)
    measure_cols = ['GlobalHorizIrr(PSP)', 'DirNormIrr', 'DiffuseHorizIrr']

    # take sum on a given time scale
    Data_sum = aggregateDf(Data, 'D', 'sum')
    # Data_sum.to_csv('data.csv')

    # get correlation between the measure columns
    print 'Correlation in {}'.format(measure_cols)
    print Data_sum[measure_cols].corr(), '\n'

    # plot Data
    # plot(measure_cols, Data_sum, '-')

    # make the dataset
    # Input, Output = createDataSets(df, split=False, window=100)

    # ARIMA
    statModel(Data_sum)


if __name__ == '__main__':
    args = vars(getParser().parse_args(sys.argv[1:]))
    Run(args)



   