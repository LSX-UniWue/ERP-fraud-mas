from sklearn.model_selection import ParameterGrid

from erp_detectors import detect_anomalies


def get_param_grid(algorithm, setting, seed):
    if algorithm == 'IsolationForest':
        if 'param_search' in setting:
            return ParameterGrid({'n_estimators': [2 ** n for n in range(4, 8)],
                                  'max_samples': [0.4, 0.6, 0.8],
                                  'max_features': [0.4, 0.6, 0.8],
                                  'bootstrap': [0],
                                  'n_jobs': [-1],
                                  'random_state': [seed]})
        elif 'baseline' in setting:
            return ParameterGrid({'n_estimators': [100], 'max_samples': [256], 'max_features': [1.0],  # pyod default
                                  'bootstrap': [0], 'n_jobs': [-1], 'random_state': [seed]})
        elif 'best' in setting:
            return ParameterGrid({'n_estimators': [128], 'max_samples': [0.8], 'max_features': [0.8],  # mas1+2 best zscore
                                  'bootstrap': [0], 'n_jobs': [-1], 'random_state': [seed]})


    elif algorithm == 'Autoencoder':
        if 'param_search' in setting:
            return ParameterGrid({'n_layers': [2, 3, 4], 'n_bottleneck': [8, 16, 32], 'learning_rate': [1e-2, 1e-3, 1e-4],
                                  'epochs': [50], 'batch_size': [2048], 'cpus': [8], 'shuffle': [True], 'verbose': [1],
                                  'device': ['cuda'], 'save_path': [None]})
        elif 'best' in setting:
            return ParameterGrid({'cpus': [8], 'n_layers': [2], 'n_bottleneck': [32], 'epochs': [50],  # mas1+2 best buckets
                                  'batch_size': [2048], 'learning_rate': [1e-2], 'shuffle': [True],
                                  'verbose': [1], 'device': ['cuda'], 'save_path': [None]})

    elif algorithm == 'OneClassSVM':
        if 'param_search' in setting:
            return ParameterGrid({"kernel": ['rbf'], 'gamma': [1e4, 1e3, 1e2, 1e1, 1e0, 1e-1, 1e-2, 1e-3, 1e-4],
                                  'tol': [1e-3], 'nu': [0.05, 0.2, 0.4, 0.6, 0.8, 0.95],
                                  'shrinking': [True], 'cache_size': [500], 'max_iter': [-1]})  # param search
        elif 'baseline' in setting:
            return ParameterGrid({"kernel": ['rbf'], 'nu': [0.5],  # pyod default
                                  'gamma': [1/36778],  # erpsim1.csv: 1/36778, erpsim2.csv: 1/37407, mas: 1/94505
                                  'tol': [1e-3], 'shrinking': [1], 'cache_size': [500], 'max_iter': [-1]})
        elif 'best' in setting:
            return ParameterGrid({"kernel": ['rbf'], 'nu': [0.05],  # mas1+2 best buckets
                                  'gamma': [1], 'tol': [1e-3], 'shrinking': [1], 'cache_size': [500], 'max_iter': [-1]})

    elif algorithm == 'pyod_AE':
        # https://pyod.readthedocs.io/en/latest/pyod.models.html#pyod.models.auto_encoder.AutoEncoder
        # TODO: please note: pyod's dataloader .item() still tries to normalize on some versions, even when disabling it...
        #  needs to set mean=np.array([0]) on data loader creations to fix their code
        return [{'preprocessing': False}]  # already processed

    else:
        raise ValueError(f"Variable algorithm was: {algorithm}")


if __name__ == '__main__':

    # Preprocessing: One of ['zscore', 'buckets']
    numeric = 'buckets'
    # Model: One of ['Autoencoder', 'OneClassSVM', 'IsolationForest', 'pyod_AE']
    algorithm = 'OneClassSVM'
    # Setting: One of ['param_search_mas1', 'param_search_mas2', 'best_erpsim1', 'best_erpsim2', 'baseline_erpsim1', 'baseline_erpsim2']
    setting = 'best_erpsim1'

    erpClassParams = {'eval_path': None,
                      'info_path': './data/erp_fraud/column_information.csv'}  # column names and dtypes
    if setting == 'param_search_mas1':
        erpClassParams['train_path'] = './data/erp_mas/normal/mas1.csv'
        erpClassParams['test_path'] = './data/erp_mas/fraud/mas1.csv'
    elif setting == 'param_search_mas2':
        erpClassParams['train_path'] = './data/erp_mas/normal/mas2.csv'
        erpClassParams['test_path'] = './data/erp_mas/fraud/mas2.csv'
    elif setting in ['best_erpsim1', 'baseline_erpsim1']:
        erpClassParams['train_path'] = './data/erp_fraud/erpsim1.csv'
        erpClassParams['test_path'] = './data/erp_fraud/erpsim1.csv'
    elif setting in ['best_erpsim2', 'baseline_erpsim2']:
        erpClassParams['train_path'] = './data/erp_fraud/erpsim2.csv'
        erpClassParams['test_path'] = './data/erp_fraud/erpsim2.csv'
    seeds = [0] if algorithm == 'OneClassSVM' else list(range(0, 5))  # 5 seeds for all except deterministic OneClassSVM

    for seed in seeds:
        param_grid = get_param_grid(algorithm=algorithm, setting=setting, seed=seed)
        for j, params in enumerate(param_grid):
            detect_anomalies(algorithm=algorithm,
                             **erpClassParams,
                             experiment_name=f'{algorithm}_erp_fraud_{str(seed)}_{str(j)}',
                             categorical='onehot',
                             numeric=numeric,
                             params=params,
                             output_scores=True,
                             seed=seed)
