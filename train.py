import argparse
import os
import pickle
import time
import shutil

from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.model_selection import train_test_split
from sdsj_feat import load_data

# use this to stop the algorithm before time limit exceeds
TIME_LIMIT = int(os.environ.get('TIME_LIMIT', 1*60))

ONEHOT_MAX_UNIQUE_VALUES = 20


if __name__ == '__main__':
    dataset = 1
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

    # train the model until time allows
    total_iter = 0 # total number of iterations to train
    iter_time = 0 # last time for one iteration
    while iter_time < TIME_LIMIT - (time.time() - start_time) - 10:
        start_iter = time.time()
        total_iter = total_iter + 100
        model_params["iterations"] = total_iter
        if args.mode == 'regression':
            model_params["loss_function"] = "RMSE"
            model = CatBoostRegressor(**model_params)
        else:
            pos_weight = df_X[df_y < 0.5].size / df_X[df_y > 0.5].size
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
