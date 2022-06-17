import torch
import sys
import os

sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))


from supervised_learning import train, test # noqa
from unimodals.common_models import GRUWithLinear, MLP # noqa
from datasets.affect.get_data import get_dataloader # noqa
from fusions.common_fusions import Concat, LowRankTensorFusion # noqa

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
_, _, testdata = get_dataloader(args.dataset_path, robust_test=False, data_type=args.dataset)

def get_range(lo, hi):
    return np.arange(hi-lo) + lo


# mosi/mosei
if args.dataset == 'mosi':
    v_in = (35, 64)
    v_out = get_range(32, 128)
    a_in = (74, 128)
    a_out = get_range(32, 128)
    l_in = (300, 512)
    l_out = 128
elif args.dataset == 'mosei':
    v_in = (713, 64)
    v_out = get_range(32, 128)
    a_in = (74, 128)
    a_out = get_range(32, 128)
    l_in = (300, 512)
    l_out = 128

# humor/sarcasm
elif args.dataset == 'humor' or args.dataset == 'sarcasm':
    d_v = [371, 512, 128]
    d_a = [81, 256, 128]
    d_l = [300, 600, 128]

total_epochs = 300


def affect_lrf(config):
    traindata, validdata, _ = get_dataloader(args.dataset_path, robust_test=False, data_type=args.dataset)
    d_v = (config["v_in"][0], config["v_in"][1], config["v_out"])
    d_a = (config["a_in"][0], config["a_in"][1], config["a_out"])
    d_l = (config["l_in"][0], config["l_in"][1], config["l_out"])
    ds = [d_v, d_a, d_l]
    d_modalities = [ds[i] for i in modalities]
    in_dim = [d[2] for d in d_modalities]
    encoders = [GRUWithLinear(d[0], d[1], d[2], dropout=True, has_padding=True).cuda() for d in d_modalities]
    head = MLP(128, 512, 1).cuda()
    fusion = LowRankTensorFusion(in_dim, 128, 32).cuda()
    saved_model = './{}_lrf_{}.pt'.format(args.dataset, ''.join(list(map(str, modalities))))
    train(encoders, fusion, head, traindata, validdata, total_epochs, task="regression", optimtype=torch.optim.AdamW, is_packed=True, lr=config["lr"], save=saved_model, weight_decay=config["weight_decay"], objective=torch.nn.L1Loss(), modalities=modalities)

    
def trial_name_string(trial):
    print("Starting trial {}".format(str(trial)))
    return str(trial)
    traintimes.append(traintime)
    mems.append(mem)


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
    affect_lrf,
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
saved_model = os.path.join(logdir, '{}_lrf_{}.pt'.format(args.dataset, ''.join(list(map(str, modalities)))))
print("Using model: {}".format(saved_model))
model = torch.load(saved_model).cuda()

test(model=model, test_dataloaders_all=testdata, dataset=args.dataset, is_packed=True, criterion=torch.nn.L1Loss(), task='posneg-classification', no_robust=True, modalities=modalities)
