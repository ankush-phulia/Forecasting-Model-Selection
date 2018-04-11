import os
import sys
import argparse
import itertools
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
# statistical models
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa import arima_model
# sklearn models
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn import preprocessing, svm, tree, neural_network, ensemble, isotonic, gaussian_process
from sklearn.metrics import mean_squared_error

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


def plot(col_names, df, combined=True):
    '''
    Plot a column of dataframe, given the flag is valid, i.e. not 99
    '''
    # check col_name is not time
    for col_name in col_names:
        if col_name not in df.columns:
            continue
        Y = df[col_name]
        plt.plot(Y, '-', label=col_name, markersize=3)

        plt.title('{} - {}'.format(
            df.index[0].strftime('%d/%m/%Y'), df.index[-1].strftime('%d/%m/%Y')))
        plt.legend(loc='upper right')
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
        flag_cols=['GHIFlag', 'DNIFlag', 'DHIFlag'],
        low_flag_bnd=2, up_flag_bnd=6):
    '''
    Remove entries with invalid flags and invalid columns
    '''
    # replace all bad flags with NaN - SERI_QC
    for measure, flag in zip(measure_cols, flag_cols):
        df.loc[df[flag] < low_flag_bnd, measure] = np.nan
        df.loc[df[flag] > up_flag_bnd, measure] = np.nan
        df.loc[df[measure] < 0, measure] = np.nan

    # remove invalid rows - all NaNs
    df.dropna(axis=0, how='all', subset=measure_cols, inplace=True)


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


def dumpSets(data_dir, train_in, train_out, test_in, test_out):
    '''
    Dump sets into files
    '''
    import pickle
    with open(os.path.join(data_dir, 'Train/train_in.pkl'), 'wb') as f:
        pickle.dump(train_in, f)
    with open(os.path.join(data_dir, 'Train/train_out.pkl'), 'wb') as f:
        pickle.dump(train_out, f)
    with open(os.path.join(data_dir, 'Test/train_in.pkl'), 'wb') as f:
        pickle.dump(test_in, f)
    with open(os.path.join(data_dir, 'Test/train_out.pkl'), 'wb') as f:
        pickle.dump(test_out, f)


def slidingWindow(df, window, imc, omc):
    '''
    Slide a window od size 'window' over data in df to get input and output
    '''
    Input, Output = [], []
    current_window = df[imc].iloc[0:window]
    for index in xrange(window, df.shape[0] - 1):
        Input.append(current_window)

        # get new obs - this is output
        new_obs = df[omc].iloc[index]
        Output.append(new_obs)

        # modify sliding window
        current_window = current_window.append(
            new_obs).drop(current_window.index[0])
    return Input, Output


def createDataSets(df, typ='continuous',
                   input_measure_cols=['DirNormIrr'],
                   input_flag_cols=['DNIFlag'],
                   output_measure_cols=['DirNormIrr'],
                   window=7, split_factor=0.2, split=True, dump_dir=''):
    '''
    Create Data set and split into train/test, dump into file
    '''

    if typ == 'cont':
        # input is continuous data of 'window' days
        Input, Output = slidingWindow(
            df, window, input_measure_cols, output_measure_cols)

    elif typ == 'date':
        # input is data for a date of the past 'window' years
        Input, Output = [], []
        for date, group in df.groupby([df.index.day, df.index.month]):
            # for each date, iterate over the years
            i, o = slidingWindow(
                group, window, input_measure_cols, output_measure_cols)
            Input += i
            Output += o

    if not split:
        return Input, Output

    # create training and testing sets
    train_in, test_in, train_out, test_out = train_test_split(
        Input, Output, test_size=split_factor, shuffle=False)

    # print information
    print 'Input  : Past {} days\' {}\nOutput : Current {}'.format(
        window, input_measure_cols, output_measure_cols)
    print 'Constructed {} samples from {} observations'.format(
        len(Input), len(df))
    print 'Training samples   : {}'.format(len(train_in))
    print 'Testing samples    : {}'.format(len(test_in))
    # print 'Validation samples : {}'.format(len(val_in))

    if len(dump_dir):
        dumpSets(dump_dir,
                 train_in, train_out, test_in, test_out)


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
        std = plt.plot(rolstd, label='Rolling Std')
        plt.legend(loc='best')
        plt.title('Rolling Mean & Standard Deviation for {}'.format(measure_col))
        plt.show()

    # Perform Dickey-Fuller test:
    dftest = adfuller(ts, autolag='AIC')
    dfoutput = pd.Series(
        dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])

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
    plt.axhline(y=-1.96 / np.sqrt(len(ts)), linestyle='--', color='gray')
    plt.axhline(y=1.96 / np.sqrt(len(ts)), linestyle='--', color='gray')
    plt.title('Autocorrelation Function for {}'.format(measure_col))
    plt.show()

    # plot pacf
    plt.plot(lag_acf)
    plt.axhline(y=0, linestyle='--', color='gray')
    plt.axhline(y=-1.96 / np.sqrt(len(ts)), linestyle='--', color='gray')
    plt.axhline(y=1.96 / np.sqrt(len(ts)), linestyle='--', color='gray')
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


def loadDumpedData(data_dir='Dumped Data'):
    '''
    Load training, testing and validation sets from dumped pickle
    '''
    import pickle
    with open(os.path.join(data_dir, 'Train/train_in.pkl')) as f:
        train_in = pickle.load(f)
    with open(os.path.join(data_dir, 'Train/train_out.pkl')) as f:
        train_out = pickle.load(f)
    with open(os.path.join(data_dir, 'Test/train_in.pkl')) as f:
        test_in = pickle.load(f)
    with open(os.path.join(data_dir, 'Test/train_out.pkl')) as f:
        test_out = pickle.load(f)
    return train_in, train_out, test_in, test_out


def crossValidateModel(train_in, train_out, model='', n=5):
    '''
    Run n-fold cross-validation for training data with various methods
    '''
    train_in = map(lambda x: x.values.T[0], train_in)
    train_out = map(lambda x: x.values.T[0], train_out)
    if model == 'SVM':
        model = make_pipeline(preprocessing.StandardScaler(), svm.SVR())
    elif model == 'ANN':
        model = make_pipeline(
            preprocessing.StandardScaler(),
            neural_network.MLPRegressor(max_iter=1))
    elif model == 'DT':
        model = make_pipeline(
            preprocessing.StandardScaler(),
            tree.DecisionTreeRegressor())
    elif model == 'GTB':
        model = make_pipeline(
            preprocessing.StandardScaler(),
            ensemble.GradientBoostingRegressor())
    elif model == 'RF':
        model = make_pipeline(
            preprocessing.StandardScaler(),
            ensemble.RandomForestRegressor())
    elif model == 'ET':
        model = make_pipeline(
            preprocessing.StandardScaler(),
            ensemble.ExtraTreesRegressor())
    elif model == 'ADA':
        model = make_pipeline(
            preprocessing.StandardScaler(),
            ensemble.AdaBoostRegressor())

    scores = cross_val_score(
        model, train_in, train_out, cv=n)
    print '{} averaged {} for {}-fold cross validation'.format(
        model, abs(sum(scores)) / n, n)


def evaluateModel(train_in, train_out, test_in, test_out, model, **kwargs):
    '''
    Evaluate a model - plot on the testing set, as well as the Mean Squared Error
    '''
    # prepare the estimator
    params = []
    if model == 'ANN':
        if len(kwargs.keys()):
            estimator = neural_network.MLPRegressor(
                hidden_layer_sizes=([kwargs['Nodes']] * kwargs['Depth']),
                solver='lbfgs', max_iter=kwargs['Iterations'],
                learning_rate_init=1e-3, learning_rate='adaptive')
        else:
            # parameter grid
            depths = [1, 2, 3, 4, 5]  # , 6, 7, 8]
            nodes = [1, 2, 5, 8, 10, 12, 15, 20, 25]  # , 30, 40, 50]
            hidden_layer_sizes_try = [[element[0]] * element[1]
                                      for element in itertools.product(*[nodes, depths])]
            params = {'max_iter': [1, 10, 100, 200],
                      'hidden_layer_sizes': hidden_layer_sizes_try}

            # grid search
            estimator = GridSearchCV(
                neural_network.MLPRegressor(
                    solver='lbfgs',
                    learning_rate_init=1e-3,
                    learning_rate='adaptive'),
                params, scoring='neg_mean_squared_error', n_jobs=4)

    elif model == 'GB':
        if len(kwargs.keys()):
            estimator = ensemble.GradientBoostingRegressor(
                n_estimators=kwargs['Estimators'],
                max_depth=kwargs['Depth'], loss='lad')
        else:
            # parameter grid
            params = {'n_estimators': [10, 25, 50, 75, 100, 125, 150],
                      'max_depth': [1, 2, 3, 5, 7, 10, 12, 15, 17, 20]}

            # grid search
            estimator = GridSearchCV(
                ensemble.GradientBoostingRegressor(loss='lad'),
                params, scoring='neg_mean_squared_error', n_jobs=4)

    elif model == 'SVM':
        if len(kwargs.keys()):
            estimator = svm.SVR(
                C=kwargs['C'], kernel=kwargs['Kernel'],
                degree=kwargs['Degree'], epsilon=kwargs['Epsilon'])
        else:
            params = {'C': [100, 200, 400, 600, 800, 1000, 1250, 1500,
                            1750, 2000, 2500, 3000, 3500, 4000, 5000],
                      'degree': [2, 3, 4, 5, 6, 7],
                      'kernel': ['poly']}
            estimator = GridSearchCV(
                svm.SVR(), params, scoring='neg_mean_squared_error', n_jobs=4)

    # make into numpy arrays
    sets = [train_in, train_out, test_in, test_out]
    sets = map(lambda y: map(lambda x: x.values.T[0], y), sets)

    # fit and predict
    estimator = make_pipeline(preprocessing.StandardScaler(), estimator)
    estimator.fit(sets[0], sets[1])
    pred_out = estimator.predict(sets[2])

    # remove negative and very large predictions
    pred_out = [(min(90000, abs(pred))) for pred in pred_out]

    # print the model details
    if len(kwargs.keys()):
        print 'Model : {}'.format(model)
        for key, value in kwargs.iteritems():
            print ' {} : {}'.format(key, value)
        print ' MSE : {}\n'.format(mean_squared_error(pred_out, sets[3]))
    else:
        estimator = estimator.named_steps['gridsearchcv']
        print 'Model : {}\n Grid Searched : \n {}\n MSE : {}'.format(
            model, estimator.best_params_, -estimator.best_score_)

    # plot the predicted and test out
    times = map(lambda x: x.name.date(), test_out)
    plt.plot(times, sets[3], 'o', label='Actual')
    plt.plot(times, pred_out, 'o', label='Predicted')
    plt.title('Daily Aggregate DNI - predicted vs actual using {}'.format(model))
    plt.legend(loc='upper right')
    plt.show()

    return pred_out


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

    # get correlation between the measure columns
    print 'Correlation in {}'.format(measure_cols)
    print Data_sum[measure_cols].corr(), '\n'

    # plot Data
    # plot(measure_cols, Data_sum, '-')

    # make the dataset & dump
    createDataSets(Data_sum, 'date',
                   split=True, window=5, dump_dir='Dumped Data Date 5')
    train_in, train_out, test_in, test_out = loadDumpedData(
        'Dumped Data Date 5')

    # try out models
    # crossValidateModel(train_in, train_out, 'SVM')
    # crossValidateModel(train_in, train_out, 'ANN')
    # crossValidateModel(train_in, train_out, 'DT')
    # crossValidateModel(train_in, train_out, 'GTB')
    # crossValidateModel(train_in, train_out, 'RF')
    # crossValidateModel(train_in, train_out, 'ET')
    # crossValidateModel(train_in, train_out, 'ADA')

    # predict using various models
    nn_args = {'Depth': 2, 'Nodes': 20, 'Iterations': 100}
    evaluateModel(train_in, train_out, test_in, test_out, 'ANN', **nn_args)
    evaluateModel(
        train_in,
        train_out,
        train_in + test_in,
        train_out + test_out,
        'ANN', **nn_args)

    gb_args = {'Depth': 10, 'Estimators': 100}
    evaluateModel(train_in, train_out, test_in, test_out, 'GB', **gb_args)
    evaluateModel(
        train_in,
        train_out,
        train_in + test_in,
        train_out + test_out,
        'GB', **gb_args)

    svm_args = {'C': 500, 'Kernel': 'linear', 'Degree': 1, 'Epsilon': 0.01}
    evaluateModel(train_in, train_out, test_in, test_out, 'SVM', **svm_args)
    evaluateModel(
        train_in,
        train_out,
        train_in + test_in,
        train_out + test_out,
        'SVM', **svm_args)

    svm_args = {'C': 1800, 'Kernel': 'poly', 'Degree': 3, 'Epsilon': 0.01}
    evaluateModel(train_in, train_out, test_in, test_out, 'SVM', **svm_args)
    evaluateModel(
        train_in,
        train_out,
        train_in + test_in,
        train_out + test_out,
        'SVM', **svm_args)


if __name__ == '__main__':
    args = vars(getParser().parse_args(sys.argv[1:]))
    Run(args)
