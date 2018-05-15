import sys
import argparse
import itertools
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
# sklearn models
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn import preprocessing, svm, tree, neural_network, ensemble, isotonic, gaussian_process
from sklearn.metrics import mean_squared_error
# other modules
import Stats
import ReadData
import PrepareData

# display setting
pd.set_option('expand_frame_repr', False)


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
        '--scale',
        default='Daily',
        help='Timescale to sum measurements - Daily or Hourly'
    )
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


def crossValidateModel(train_in, train_out, model_name='', n=5):
    '''
    Run n-fold cross-validation for training data with various methods
    '''
    train_in = map(lambda x: x.values.T[0], train_in)
    train_out = map(lambda x: x.values.T[0], train_out)
    if model_name == 'SVM':
        model = make_pipeline(preprocessing.StandardScaler(), svm.SVR())
    elif model_name == 'ANN':
        model = make_pipeline(
            preprocessing.StandardScaler(),
            neural_network.MLPRegressor(max_iter=1))
    elif model_name == 'DT':
        model = make_pipeline(
            preprocessing.StandardScaler(),
            tree.DecisionTreeRegressor())
    elif model_name == 'GTB':
        model = make_pipeline(
            preprocessing.StandardScaler(),
            ensemble.GradientBoostingRegressor())
    elif model_name == 'Random Forest':
        model = make_pipeline(
            preprocessing.StandardScaler(),
            ensemble.RandomForestRegressor())
    elif model_name == 'Extra Trees':
        model = make_pipeline(
            preprocessing.StandardScaler(),
            ensemble.ExtraTreesRegressor())
    elif model_name == 'ADABoost':
        model = make_pipeline(
            preprocessing.StandardScaler(),
            ensemble.AdaBoostRegressor())

    scores = cross_val_score(
        model, train_in, train_out, cv=n)
    print '{} averaged {} for {}-fold cross validation'.format(
        model_name, (sum(scores)) / n, n)


def evaluateModel(train_in, train_out, test_in,
                  test_out, model, save_train=False, save_all=True,
                  **kwargs):
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

    elif model == 'GradientBoost':
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
                degree=kwargs['Depth'], epsilon=kwargs['Epsilon'])
        else:
            params = {'C': [100, 200, 400, 600, 800, 1000, 1250, 1500,
                            1750, 2000, 2500, 3000, 3500, 4000, 5000],
                      'degree': [2, 3, 4, 5, 6, 7],
                      'kernel': ['poly']}
            estimator = GridSearchCV(
                svm.SVR(), params, scoring='neg_mean_squared_error', n_jobs=4)

    elif model == 'ADABoost':
        if len(kwargs.keys()):
            estimator = ensemble.AdaBoostRegressor(
                n_estimators=kwargs['Estimators'])
        else:
            # parameter grid
            params = {'n_estimators': [10, 25, 50, 75, 100, 125, 150]}

            # grid search
            estimator = GridSearchCV(
                ensemble.AdaBoostRegressor(),
                params, scoring='neg_mean_squared_error', n_jobs=4)

    elif model == 'Extra Trees':
        if len(kwargs.keys()):
            estimator = ensemble.ExtraTreesRegressor(
                n_estimators=kwargs['Estimators'],
                max_depth=kwargs['Depth'])
        else:
            # parameter grid
            params = {'n_estimators': [25, 50, 75, 100, 125, 150, 175, 200],
                      'max_depth': [10, 15, 20, 25, 30, 35, 40, 45, 50]}

            # grid search
            estimator = GridSearchCV(
                ensemble.ExtraTreesRegressor(),
                params, scoring='neg_mean_squared_error', n_jobs=4)

    elif model == 'Random Forest':
        if len(kwargs.keys()):
            estimator = ensemble.RandomForestRegressor(
                n_estimators=kwargs['Estimators'],
                max_depth=kwargs['Depth'])
        else:
            # parameter grid
            params = {'n_estimators': [25, 50, 75, 100, 125, 150, 175, 200],
                      'max_depth': [10, 15, 20, 25, 30, 35, 40, 45, 50]}

            # grid search
            estimator = GridSearchCV(
                ensemble.RandomForestRegressor(),
                params, scoring='neg_mean_squared_error', n_jobs=4)

    # make into numpy arrays
    sets = [train_in, train_out, test_in, test_out]
    sets = map(lambda y: map(lambda x: x.values.T[0], y), sets)

    # fit and predict
    estimator = make_pipeline(preprocessing.StandardScaler(), estimator)
    estimator.fit(sets[0], sets[1])
    pred_out = estimator.predict(sets[2])
    pred_out_total = estimator.predict(sets[0] + sets[2])

    # remove negative and very large predictions
    # pred_out = [(max(0, pred)) for pred in pred_out]
    pred_out = [(min(85000, abs(pred))) for pred in pred_out]
    pred_out_total = [(min(85000, abs(pred))) for pred in pred_out_total]
    mse_test = mean_squared_error(pred_out, sets[3])
    mse_overall = mean_squared_error(pred_out_total, sets[1] + sets[3])

    # print the model details
    if len(kwargs.keys()):
        print 'Model : {}'.format(model)
        for key, value in kwargs.iteritems():
            print ' {} : {}'.format(key, value)
        print ' MSE on Test Set: {}'.format(mse_test)
        print ' MSE Overall: {}\n'.format(mse_overall)
    else:
        estimator = estimator.named_steps['gridsearchcv']
        print 'Model : {}\n Grid Searched : \n {}\n MSE : {}'.format(
            model, estimator.best_params_, -estimator.best_score_)

    # figure out the scale based on data spacing
    temp = test_out[0].name
    if temp.hour + temp.minute + temp.second == 0:
        scale = 'Daily'
    else:
        scale = 'Hourly'

    # plot the predicted and test out
    times = map(lambda x: x.name.date(), test_out)
    plt.plot(times, sets[3], 'o', label='Actual', markersize=3)
    plt.plot(times, pred_out, 'o', label='Predicted', markersize=3)
    plt.gca().set_ylabel('Agg. DNI')
    plt.gca().set_xlabel('MSE = {}'.format(mse_test))
    plt.title(
        '{} Agg. DNI - predicted vs actual using {} - Test Set'.format(scale, model))
    plt.legend(loc='upper right')
    # plt.show()
    if save_train:
        plt.savefig('{}_{}_{}.png'.format(model, len(test_in), kwargs['Depth']))
    plt.close()

    times = map(lambda x: x.name.date(), train_out + test_out)
    plt.plot(times, sets[1] + sets[3], 'o', label='Actual', markersize=3)
    plt.plot(times, pred_out_total, 'o', label='Predicted', markersize=3)
    plt.gca().set_ylabel('Agg. DNI')
    plt.gca().set_xlabel('MSE = {}'.format(mse_overall))
    plt.title(
        '{} Agg. DNI - predicted vs actual using {} - Overall'.format(scale, model))
    plt.legend(loc='upper right')
    # plt.show()
    if save_all:
        plt.savefig(
            '{}_{}_{}.png'.format(model, len(train_in + test_in), kwargs['Depth']))
    plt.close()

    return pred_out


def runModels(train_in, train_out, test_in, test_out, scale):
    '''
    A collection of (relatively) tuned models for different time scales
    '''
    if scale == 'Date':
        nn_args =   {'Depth': 2, 'Nodes': 20, 'Iterations': 100}
        gb_args =   {'Depth': 15, 'Estimators': 200}
        gb_args2 =  {'Depth': 10, 'Estimators': 200}
        svm_args =  {'Depth': 1, 'C': 500, 'Kernel': 'linear', 'Epsilon': 0.01}
        svm_args2 = {'Depth': 3, 'C': 1800, 'Kernel': 'poly', 'Epsilon': 0.01}
        ada_args =  {'Depth': 10, 'Estimators': 100}
        et_args =   {'Depth': 10, 'Estimators': 200}
        et_args2 =  {'Depth': 20, 'Estimators': 100}
        rf_args =   {'Depth': 50, 'Estimators': 200}

    elif scale == 'Hour':
        nn_args =   {'Depth': 2, 'Nodes': 30, 'Iterations': 10000}
        gb_args =   {'Depth': 15, 'Estimators': 200}
        gb_args2 =  {'Depth': 25, 'Estimators': 100}
        svm_args =  {'Depth': 1, 'C': 100, 'Kernel': 'linear', 'Epsilon': 0.01}
        # svm_args2 = {'C': 1500, 'Kernel': 'poly', 'Epsilon': 0.01}
        svm_args2 = {'Depth': 3, 'C': 200, 'Kernel': 'poly', 'Epsilon': 0.01}
        ada_args =  {'Depth': 10, 'Estimators': 150}
        et_args =   {'Depth': 20, 'Estimators': 200}
        rf_args =   {'Depth': 40, 'Estimators': 300}

    evaluateModel(train_in, train_out, test_in, test_out, 'ANN', **nn_args)

    evaluateModel(
        train_in, train_out, test_in, test_out,
        'GradientBoost', **gb_args)
    evaluateModel(
        train_in, train_out, test_in, test_out,
        'GradientBoost', **gb_args2)

    evaluateModel(train_in, train_out, test_in, test_out, 'SVM', **svm_args)
    evaluateModel(train_in, train_out, test_in, test_out, 'SVM', **svm_args2)

    evaluateModel(
        train_in, train_out, test_in, test_out,
        'ADABoost', **ada_args)

    evaluateModel(
        train_in, train_out, test_in, test_out,
        'Extra Trees', **et_args)

    evaluateModel(
        train_in, train_out, test_in, test_out,
        'Random Forest', **rf_args)


def Run(args):
    cid = plt.gcf().canvas.mpl_connect('key_press_event', closePlot)

    # get the combined data - cleaned
    Data = ReadData.Run(args)
    measure_cols = ['DNI', 'GHI', 'DHI']

    # sort out the scale info
    scale_map = {'Hourly':'Hour', 'Daily':'Date'}
    scale = scale_map[args['scale']]
    window = 5

    # take sum on a given time scale
    Data_sum = PrepareData.aggregateDf(Data, scale[0], 'sum')

    # get correlation between the measure columns
    print 'Correlation in {}'.format(measure_cols)
    print Data_sum.corr(), '\n'

    # plot Data
    plot(['DNI'], Data_sum, '-')

    # make the dataset & dump
    # PrepareData.createDataSets(Data_sum, scale,
    #                 input_measure_cols=['DNI'], output_measure_cols=['DNI'],
    #                 split=True, window=window, 
    #                 dump_dir='Dumped Dataset/Suny/{} {}'.format(scale, window))
    train_in, train_out, test_in, test_out = PrepareData.loadDumpedData(
        'Dumped Dataset/Suny/{} {}'.format(scale, window))

    runModels(train_in, train_out, test_in, test_out, scale)

    


if __name__ == '__main__':
    args = vars(getParser().parse_args(sys.argv[1:]))
    Run(args)
