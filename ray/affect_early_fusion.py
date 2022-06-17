import torch
import sys
import os
sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))
from unimodals.common_models import GRU, MLP, Sequential, Identity  # noqa
from supervised_learning import train, test  # noqa
from datasets.affect.get_data import get_dataloader  # noqa
from fusions.common_fusions import ConcatEarly  # noqa

import numpy as np
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
_, _, testdata = get_dataloader(args.dataset_path, robust_test=False, max_pad=True, data_type=args.dataset, max_seq_len=50)

# mosi/mosei
if args.dataset == 'mosi':
    d_v = (35, 70)
    d_a = (74, 200)
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
    
config = [d_v, d_a, d_l]
d_modalities = [config[i] for i in modalities]
out_dim = sum([d[0] for d in d_modalities])

total_epochs = 300


def affect_early_fusion(config):
    traindata, validdata, _ = get_dataloader(args.dataset_path, robust_test=False, max_pad=True, data_type=args.dataset, max_seq_len=50)
    encoders = [Identity().cuda() for _ in modalities]
    head = Sequential(GRU(out_dim, 512, dropout=True, has_padding=False, batch_first=True, last_only=True), MLP(512, 512, 1)).cuda()
    fusion = ConcatEarly().cuda()
    saved_model = './{}_ef_{}.pt'.format(args.dataset, ''.join(list(map(str, modalities))))
    train(encoders, fusion, head, traindata, validdata, total_epochs, task="regression", optimtype=torch.optim.AdamW, 
      is_packed=False, lr=config["lr"], save=saved_model, weight_decay=config["weight_decay"], objective=torch.nn.L1Loss(), modalities=modalities)

    
def trial_name_string(trial):
    print("Starting trial {}".format(str(trial)))
    return str(trial)
    traintimes.append(traintime)
    mems.append(mem)


search_space = {
    "lr": tune.loguniform(10**-3.1, 10**-2.9),
    "weight_decay": tune.loguniform(0.009, 0.011),
    # "lr": tune.choice([1e-3]),
    # "weight_decay": tune.choice([0.01])
}
hyperopt_search = HyperOptSearch(metric="valid_loss", mode="min")
analysis = tune.run(
    affect_early_fusion,
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
saved_model = os.path.join(logdir, '{}_ef_{}.pt'.format(args.dataset, ''.join(list(map(str, modalities)))))
print("Using model: {}".format(saved_model))
model = torch.load(saved_model).cuda()

test(model, testdata, 'affect', is_packed=False, criterion=torch.nn.L1Loss(), task="posneg-classification", no_robust=True, modalities=modalities)
