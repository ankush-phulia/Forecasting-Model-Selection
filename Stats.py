import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa import arima_model


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


def statModel(df, measure_cols=['GHI']):
    '''
    Model Time series data with statistical models
    '''
    for col in measure_cols:
        checkStationarity(df, col)
        plotACF_PACF(df, col)
        ts = df[col]
        p, q, d = 80, 80, 0
        model = arima_model.ARIMA(ts, order=(p, d, q)).fit()
        plt.plot(ts)
        plt.plot(model.fittedvalues, color='red')
        plt.title('RSS : {}'.format(sum((model.fittedvalues - ts)**2)))
