import os
import sys
import argparse
import numpy as np
import pandas as pd


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
    parser.add_argument(
        '--save',
        default=False,
        help='Whether to save the combined dataframe to csv'
    )
    return parser


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
    # relevant column names for the site
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


def Run(args):
    # read data to make list of dataframes
    dfs, measure_cols, flag_cols = readData(
        args['data_dir'], args['data'], args['years'], args['start_month'], args['num_months'])

    # if data is already in combined/cleaned form
    if len(dfs) == 1 and len(args['data']):
        return dfs[0]

    # clean-up for each one
    for df in dfs:
        cleanupDf(df, measure_cols, flag_cols)

    # put them all together
    Data = pd.concat(dfs)

    if args.get('save', False):
        Data.to_csv('{}.csv'.format(args['data_dir'].split('/')[-1]))

    return Data


if __name__ == '__main__':
    args = vars(getParser().parse_args(sys.argv[1:]))
    Run(args)
