import sys
import os
sys.path.insert(1,os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))
from supervised_learning import train, test
from unimodals.common_models import GRU, MLP
from datasets.affect.get_data import get_dataloader
from fusions.common_fusions import Concat
import numpy as np
import torch

import argparse
import json

from ray import tune
from ray.tune.schedulers import ASHAScheduler
from hyperopt import hp
from ray.tune.suggest.hyperopt import HyperOptSearch


parser = argparse.ArgumentParser()
parser.add_argument("--modalities", default='[0,1,2]', type=str)
parser.add_argument("--dataset", default='mosi', type=str)
parser.add_argument("--dataset-path", default='/usr0/home/yuncheng/MultiBench/data/mosi_raw.pkl', type=str)
args = parser.parse_args()

modalities = json.loads(args.modalities)

def affect_train(config):
    # mosi_data.pkl, mosei_senti_data.pkl
    # mosi_raw.pkl, mosei_raw.pkl, sarcasm.pkl, humor.pkl
    # raw_path: mosi.hdf5, mosei.hdf5, sarcasm_raw_text.pkl, humor_raw_text.pkl
    traindata, validdata, test_robust = get_dataloader(args.dataset_path, robust_test=False, data_type=args.dataset)

    # mosi/mosei
    if args.dataset == 'mosi':
        d_v = (35, 600)
        d_a = (74, 600)
        d_l = (300, 600)
    elif args.dataset == 'mosei':
        d_v = (713, 70)
        d_a = (74, 200)
        d_l = (300, 600)

    # humor/sarcasm
    elif args.dataset == 'humor' or args.dataset == 'sarcasm':
        d_v = (371, 600)
        d_a = (81, 600)
        d_l = (300, 600)

    ds = [d_v, d_a, d_l]
    d_modalities = [ds[i] for i in modalities]
    out_dim = sum([d[1] for d in d_modalities])
    encoders = [GRU(d[0], d[1], dropout=True, has_padding=True, batch_first=True).cuda() for d in d_modalities]
    head = MLP(out_dim, 512, 1).cuda()

    fusion = Concat().cuda()

    saved_model = './{}_lf_{}.pt'.format(args.dataset, ''.join(list(map(str, modalities))))

    train(encoders, fusion, head, traindata, validdata, 100, task="regression", optimtype=torch.optim.AdamW,
      early_stop=False, is_packed=True, lr=config["lr"], save=saved_model, weight_decay=config["weight_decay"], objective=torch.nn.L1Loss(), modalities=modalities)


search_space = {
    "lr": hp.loguniform("lr", 1e-5, 0.1),
    "weight_decay": tune.uniform(0.01, 0.1)
}
hyperopt_search = HyperOptSearch(metric="mean_accuracy", mode="max")
analysis = tune.run(
    affect_train,
    num_samples=20,
    search_alg=hyperopt_search,
    scheduler=ASHAScheduler(metric="mean_accuracy", mode="max"),
    config=search_space,
    resources_per_trial={
            "cpu": 1,
            "gpu": 0.5,
        }
)

dfs = analysis.trial_dataframes
ax = None
for d in dfs.values():
    ax = d.mean_accuracy.plot(ax=ax, legend=False)

print("Testing:")
logdir = analysis.get_best_logdir("mean_accuracy", mode="max")
saved_model = os.path.join(logdir, '{}_lf_{}.pt'.format(args.dataset, ''.join(list(map(str, modalities)))))
model = torch.load(saved_model).cuda()

test(model=model, test_dataloaders_all=test_robust, dataset=args.dataset, is_packed=True, criterion=torch.nn.L1Loss(), task='posneg-classification', no_robust=True, modalities=modalities)
