import pandas as pd
import numpy as np
import argparse
import os
import random
from autogluon.tabular import TabularPredictor, TabularDataset
from autogluon.tabular.configs.hyperparameter_configs import get_hyperparameter_config
import torch as th
from sklearn.model_selection import train_test_split
try:
    from autogluon.multimodal.utils import download
except:
    from autogluon.multimodal.utils.download import download

import warnings
warnings.filterwarnings('ignore')


def get_parser():
    parser = argparse.ArgumentParser(
        description='The Basic Example of XTab on adult dataset.')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--max_epochs', type=int, default=3)
    parser.add_argument('--pretrained_ckpts', default='./pretrain3.ckpt')
    return parser


def data_loader(path="./dataset/", ):
    name = INFO["name"]
    full_path = os.path.join(path, name)
    if os.path.exists(full_path):
        print(f"Existing dataset: {name}")
    else:
        print(f"Dataset not exist. Start downloading: {name}")
        download(INFO["url"], path=full_path, sha1_hash=INFO["sha1sum"])
    df = pd.read_csv(full_path, sep='\t')
    return df


def train(args):
    df_train = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/Inc/train.csv')
    df_test = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv')
    label = "class"

    hyperparameters = {} 
    hyperparameters['FT_TRANSFORMER'] = {
        "env.per_gpu_batch_size": args.batch_size,
        "env.num_workers": 0,
        "env.num_workers_evaluation": 0,
        "optimization.max_epochs": args.max_epochs,
        'finetune_on': args.pretrained_ckpts,
    }
    print(hyperparameters)

    predictor = TabularPredictor(label=label,
                                 eval_metric="roc_auc",
                                 )

    df_train = df_train.dropna(subset=[label])
    df_test = df_test.dropna(subset=[label])

    predictor.fit(
        train_data=df_train,
        hyperparameters=hyperparameters,
        time_limit=60,
        keep_only_best = True,
        fit_weighted_ensemble = False,
    )

    probabilities = predictor.predict(df_test, as_pandas=True)
    leaderboard = predictor.leaderboard(df_test)
    leaderboard.to_csv("./leaderboard.csv")
    return


def set_seed(seed):
    th.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    set_seed(args.seed)
    train(args)