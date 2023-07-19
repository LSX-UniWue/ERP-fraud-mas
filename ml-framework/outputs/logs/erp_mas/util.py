
import glob
from pathlib import Path
import numpy as np
import pandas as pd


if __name__ == '__main__':
    """Join logs from multiple runs into one file."""

    dataset = 'fraud3'
    approach = 'SVM'
    prep = 'buckets'
    type = 'param_search'

    # set hyperparameter columns
    if approach == 'AE':
        if type == 'baseline':
            params = ['preprocessing']
        else:
            params = ['learning_rate', 'n_bottleneck', 'n_layers', 'batch_size', 'epochs', 'n_inputs']
    elif approach == 'SVM':
        params = ['kernel', 'gamma', 'nu']
    elif approach == 'IF':
        params = ['n_estimators', 'max_samples', 'max_features']
    else:
        raise ValueError('unknown approach: ' + approach)

    files = []
    for file_path in glob.glob(f'./{dataset}/{approach}/{prep}/{type}/*.csv'):
        if not file_path.endswith('scores.csv'):
            files.append(pd.read_csv(file_path))
    df = pd.concat(files, ignore_index=True)

    score_cols = ['auc_roc_eval', 'auc_pr_eval', 'auc_roc_test', 'auc_pr_test'] if 'auc_roc_eval' in df.columns else ['auc_roc_test', 'auc_pr_test']
    means = df.groupby(params).mean().reset_index()
    stds = df.groupby(params).std().reset_index()
    for col in score_cols:
        means[col + '_std'] = stds[col]

    score_cols = [[score, score + '_std'] for score in score_cols]
    score_cols = [item for sublist in score_cols for item in sublist]

    df = means[[*params, *score_cols]]
    df.to_csv(f'summary_{dataset}_{approach}_{prep}_{type}.csv', index=False)
