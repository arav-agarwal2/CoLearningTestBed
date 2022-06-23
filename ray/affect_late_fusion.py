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
_, _, testdata = get_dataloader(args.dataset_path, robust_test=False, data_type=args.dataset)


def get_range(lo, hi):
    return np.arange(hi-lo) + lo

# mosi/mosei
if args.dataset == 'mosi':
    v_in = 35
    v_out = get_range(70, 600)
    a_in = 74
    a_out = get_range(200, 600)
    l_in = 300
    l_out = 600
elif args.dataset == 'mosei':
    v_in = 713
    v_out = get_range(70, 600)
    a_in = 74
    a_out = get_range(200, 600)
    l_in = 300
    l_out = 600

# humor/sarcasm
elif args.dataset == 'humor' or args.dataset == 'sarcasm':
    v_in = 371
    a_in = 81
    l_in = 300

total_epochs = 300


def affect_late_fusion(config):
    traindata, validdata, _ = get_dataloader(args.dataset_path, robust_test=False, data_type=args.dataset)
    d_v = (config["v_in"], config["v_out"])
    d_a = (config["a_in"], config["a_out"])
    d_l = (config["l_in"], config["l_out"])
    ds = [d_v, d_a, d_l]
    d_modalities = [ds[i] for i in modalities]
    out_dim = sum([d[1] for d in d_modalities])
    encoders = [GRU(d[0], d[1], dropout=True, has_padding=True, batch_first=True).cuda() for d in d_modalities]
    head = MLP(out_dim, 512, 1).cuda()
    fusion = Concat().cuda()
    saved_model = './{}_lf_{}.pt'.format(args.dataset, ''.join(list(map(str, modalities))))
    train(encoders, fusion, head, traindata, validdata, total_epochs, task="regression", optimtype=torch.optim.AdamW,
      early_stop=False, is_packed=True, lr=config["lr"], save=saved_model, weight_decay=config["weight_decay"], objective=torch.nn.L1Loss(), modalities=modalities)


def trial_name_string(trial):
    print("Starting trial {}".format(str(trial)))
    return str(trial)

search_space = {
    "lr": tune.loguniform(10**-3.1, 10**-2.9),
    "weight_decay": tune.loguniform(0.009, 0.011),
    "v_in": tune.choice([v_in]),
    "v_out": tune.choice(v_out),
    "a_in": tune.choice([a_in]),
    "a_out": tune.choice(a_out),
    "l_in": tune.choice([l_in]),
    "l_out": tune.choice([l_out]),
    # "lr": tune.choice([1e-3]),
    # "weight_decay": tune.choice([0.01])
}
hyperopt_search = HyperOptSearch(metric="valid_loss", mode="min")
analysis = tune.run(
    affect_late_fusion,
    num_samples=20,
    search_alg=hyperopt_search,
    scheduler=ASHAScheduler(metric="valid_loss", mode="min", grace_period=30, max_t=total_epochs),
    config=search_space,
    resources_per_trial={
            "cpu": 1,
            "gpu": 0.5,
        },
    local_dir="./ray_results",
    trial_name_creator=trial_name_string
)

print("Testing:")
logdir = analysis.get_best_logdir("valid_loss", mode="min")
saved_model = os.path.join(logdir, '{}_lf_{}.pt'.format(args.dataset, ''.join(list(map(str, modalities)))))
print("Using model: {}".format(saved_model))
model = torch.load(saved_model).cuda()

test(model=model, test_dataloaders_all=testdata, dataset=args.dataset, is_packed=True, criterion=torch.nn.L1Loss(), task='posneg-classification', no_robust=True, modalities=modalities)