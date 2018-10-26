import argparse
import os
import pickle
import time
import shutil

from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.model_selection import train_test_split
from sdsj_feat import load_data

from hyperopt import hp, tpe, STATUS_OK, space_eval, Trials
import hyperopt
from catboost import FeaturesData, Pool

import pandas as pd
import numpy as np

# use this to stop the algorithm before time limit exceeds
TIME_LIMIT = int(os.environ.get('TIME_LIMIT', 5*60))

ONEHOT_MAX_UNIQUE_VALUES = 20

def get_best_params(X, y, mode):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.5)
    train_data = Pool(data=X_train, label=y_train)
    valid_data = Pool(data=X_val, label=y_val)

    space = {
        "iterations": hp.choice("iterations", range(100, 1000, 50)),
        "learning_rate": hp.uniform("learning_rate", 0.01, 0.05),
        "depth": hp.choice("max_depth", range(1, 10)),
        "bagging_temperature": hp.choice("bagging_temperature", [0, 1, 2]),
        "random_strength": hp.choice("random_strength", [0.5, 1, 1.5, 2]),
    }

    def objective(hyperparams):
        print('Hypers:', hyperparams, end='')
        model = CatBoostClassifier(**hyperparams) if mode == 'classification' else CatBoostRegressor(**hyperparams)
        model = model.fit(train_data, logging_level='Silent', eval_set=valid_data)

        errors = pd.read_csv('catboost_info/test_error.tsv', sep='\t')
        loss = errors.columns[-1]
        score = errors.min()[loss]
        print(' --> Score: {:.2f}'.format(score))

        return {'loss': score, 'status': STATUS_OK}

    trials = Trials()
    best = hyperopt.fmin(fn=objective, space=space, trials=trials, algo=tpe.suggest, max_evals=10,
                         rstate=np.random.RandomState(1))

    hyperparams = space_eval(space, best)
    return hyperparams

if __name__ == '__main__':
    dataset = 2
    tasktype = 'r'
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-csv', default='../sberbank_data/check_{}_{}/train.csv'.format(dataset, tasktype))
    parser.add_argument('--model-dir', default='check_{}_{}_model/'.format(dataset, tasktype))
    parser.add_argument('--mode', choices=['classification', 'regression'],
                        default = 'regression' if tasktype == 'r' else 'classification')
    args = parser.parse_args()

    if not os.path.exists(args.model_dir):
        os.mkdir(args.model_dir)

    start_time = time.time()

    print('Dataset:', args.train_csv)
    df_X, df_y, model_config, _ = load_data(args.train_csv)

    model_config['mode'] = args.mode

    model_config_filename = os.path.join(args.model_dir, 'model_config.pkl')
    with open(model_config_filename, 'wb') as fout:
        pickle.dump(model_config, fout, protocol=pickle.HIGHEST_PROTOCOL)

    # eval dataset during the training
    size = min(df_X.shape[0] // 10, 1000)
    X_train, X_eval, y_train, y_eval = train_test_split(df_X, df_y, test_size=size)

    train_dir = 'catboost_info/'
    if os.path.exists(train_dir):
        shutil.rmtree(train_dir)

    model_params = {"iterations": 0,
                    "one_hot_max_size": 10,
                    "nan_mode": 'Min',
                    "depth": 6,
                    "used_ram_limit":'512mb',
                    "loss_function": 'Logloss',
                    "save_snapshot": True,
                    # "custom_metric": 'AUC:hints=skip_train~false',
                    # "metric_period": 20,
                    "train_dir": train_dir,
                    }

    idx = np.random.choice(range(df_X.shape[0]), 100)
    X_sample, y_sample = df_X.loc[idx], df_y[idx]
    time2params = time.time()
    hp_params = get_best_params(X_sample, y_sample, model_config['mode'])
    print('Time to get parameters:', time.time() - time2params)
    model_params.update(hp_params)
    print('Final model params:', model_params)

    # train the model until time allows
    total_iter = 0 # total number of iterations to train
    iter_time = 0 # last time for one iteration
    N = model_params['iterations']
    while iter_time < TIME_LIMIT - (time.time() - start_time) - 10 \
            and total_iter < N:
        start_iter = time.time()
        total_iter = total_iter + 100
        model_params["iterations"] = total_iter
        if args.mode == 'regression':
            model_params["loss_function"] = "RMSE"
            model = CatBoostRegressor(**model_params)
        else:
            pos_weight = df_X[df_y < 0.5].shape[0] / df_X[df_y > 0.5].shape[0]
            model_params["scale_pos_weight"] = pos_weight
            model = CatBoostClassifier(**model_params)

        model.fit(df_X, df_y, model_config['cat_indices'],
                  logging_level='Silent',
                  use_best_model=True,
                  # early_stopping_rounds=10,
                  eval_set=[(X_eval, y_eval)],
                  )
        model.save_model(os.path.join(args.model_dir, 'model.catboost'))
        iter_time = time.time() - start_iter
        print('Time per iteration: {}'.format(iter_time))



    # model_config['model'] = model
