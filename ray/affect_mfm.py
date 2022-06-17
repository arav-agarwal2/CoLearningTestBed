import torch
import sys
import os

sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))

from unimodals.MVAE import TSEncoder, TSDecoder # noqa
from utils.helper_modules import Sequential2 # noqa
from objective_functions.objectives_for_supervised_learning import MFM_objective # noqa
from torch import nn # noqa
from unimodals.common_models import MLP # noqa
from supervised_learning import train, test # noqa
from datasets.affect.get_data import get_dataloader # noqa
from fusions.common_fusions import Concat # noqa

import numpy as np
import argparse
import json

from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.suggest.hyperopt import HyperOptSearch


parser = argparse.ArgumentParser()
parser.add_argument("--modalities", default='[0,1,2]', type=str)
parser.add_argument("--dataset", default='mosi', type=str)
parser.add_argument("--dataset-path", default='/usr0/home/yuncheng/MultiBench/data/mosi_raw.pkl', type=str)
args = parser.parse_args()

classes = 2
n_latent = 256
timestep = 50

modalities = json.loads(args.modalities)
_, _, testdata = get_dataloader(args.dataset_path, task='classification', robust_test=False, max_pad=True, max_seq_len=timestep, data_type=args.dataset)

def get_range(lo, hi):
    return np.arange(hi-lo) + lo

# mosi/mosei
if args.dataset == 'mosi':
    dim_0 = 35
    dim_1 = 74
    dim_2 = 300
elif args.dataset == 'mosei':
    dim_0 = 713
    dim_1 = 74
    dim_2 = 300

# humor/sarcasm
elif args.dataset == 'humor' or args.dataset == 'sarcasm':
    dim_0 = 371
    dim_1 = 81
    dim_2 = 300

ds = [dim_0, dim_1, dim_2]
d_modalities = [ds[i] for i in modalities]

total_epochs = 300


def affect_mfm(config):
    if "n_latent" in config:
        n_latent = config["n_latent"]//2*2
    traindata, validdata, _ = get_dataloader(args.dataset_path, task='classification', robust_test=False, max_pad=True, max_seq_len=timestep, data_type=args.dataset)
    encoders = [TSEncoder(d, 30, n_latent, timestep, returnvar=False).cuda() for d in d_modalities]
    decoders = [TSDecoder(d, 30, n_latent, timestep).cuda() for d in d_modalities]
    fuse = Sequential2(Concat(), MLP(len(modalities)*n_latent, n_latent, n_latent//2)).cuda()
    intermediates = [MLP(n_latent, n_latent//2, n_latent//2).cuda() for _ in modalities]
    head = MLP(n_latent//2, 20, classes).cuda()
    argsdict = {'decoders': decoders, 'intermediates': intermediates}
    additional_modules = decoders+intermediates
    objective = MFM_objective(2.0, [torch.nn.MSELoss(), torch.nn.MSELoss(), torch.nn.MSELoss()], [1.0, 1.0, 1.0])
    saved_model = './{}_mfm_{}.pt'.format(args.dataset, ''.join(list(map(str, modalities))))
    train(encoders, fuse, head, traindata, validdata, total_epochs, additional_modules, objective=objective, objective_args_dict=argsdict, lr=config["lr"], save=saved_model, modalities=modalities)

    
def trial_name_string(trial):
    print("Starting trial {}".format(str(trial)))
    return str(trial)


search_space = {
    "lr": tune.loguniform(10**-3.1, 10**-2.9),
    "n_latent": tune.choice(get_range(256, 600)),
}
hyperopt_search = HyperOptSearch(metric="valid_loss", mode="min")
analysis = tune.run(
    affect_mfm,
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
saved_model = os.path.join(logdir, '{}_mfm_{}.pt'.format(args.dataset, ''.join(list(map(str, modalities)))))
print("Using model: {}".format(saved_model))
model = torch.load(saved_model).cuda()

test(model=model, test_dataloaders_all=testdata, dataset=args.dataset, is_packed=False, no_robust=True, modalities=modalities)
