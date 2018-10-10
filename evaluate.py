from sklearn.metrics import mean_squared_error as mse, roc_auc_score as roc
import pandas as pd
from catboost import CatBoostClassifier
import argparse
import os
import time

if __name__ == '__main__':
    dataset = 1
    tasktype = 'r'
    parser = argparse.ArgumentParser()
    parser.add_argument('--true-label', type=argparse.FileType('r'),
                        default = '../sberbank_data/check_{}_{}/test-target.csv'.format(dataset, tasktype))
    parser.add_argument('--predicted-label', type=argparse.FileType('r'),
                        default = 'check_{}_{}_model/predictions.csv'.format(dataset, tasktype))
    parser.add_argument('--evaluation-csv')
    parser.add_argument('--mode', choices=['classification', 'regression'],
                        default='regression' if tasktype == 'r' else 'classification')
    args = parser.parse_args()

    start_time = time.time()


    test_path = args.true_label
    pred_path = args.predicted_label

    df_test = pd.read_csv(test_path)
    df_pred = pd.read_csv(pred_path)

    df = df_test.merge(df_pred, on = 'line_id')

    if args.mode == 'regression':
        error = mse(df['target'], df['prediction']) ** .5
    elif args.mode == 'classification':
        error = roc(df['target'], df['prediction'])

    if args.evaluation_csv:
        print(args.evaluation_csv)
        with open(args.evaluation_csv, 'a+') as f:
            f.write("{:.6f},".format(error))

    print(f"{args.mode} error {error}.\ntime spent: {time.time() - start_time}")