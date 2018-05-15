import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def aggregateDf(df, col, operation='avg',
                possible_cols=['Y', 'M', 'D', 'h']):
    '''
    Groups all the observations of df by the cols, and aggregates over that time span
    '''
    # TODO - hack to handle hour
    if col == 'H': col = 'h'

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
                   input_measure_cols=['DNI', 'GHI'],
                   output_measure_cols=['DNI'],
                   window=7, split_factor=0.2, split=True, dump_dir=''):
    '''
    Create Data set and split into train/test, dump into file
    '''
    if typ == 'Cont':
        # input is continuous data of 'window' days
        Input, Output = slidingWindow(
            df, window, input_measure_cols, output_measure_cols)
    else:
        # input is data for a date of the past 'window' years
        Input, Output = [], []
        if typ == 'Date':
            group_cols = [df.index.day, df.index.month]
        if typ == 'Hour':
            group_cols = [df.index.hour, df.index.day, df.index.month]

        for date, group in df.groupby(group_cols):
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
