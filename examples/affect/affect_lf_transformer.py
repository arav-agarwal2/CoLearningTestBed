import torch
import sys
import os

sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))

from private_test_scripts.all_in_one import all_in_one_train # noqa
from training_structures.Supervised_Learning import train, test # noqa
from unimodals.common_models import Transformer, MLP # noqa
from datasets.affect.get_data import get_dataloader # noqa
from fusions.common_fusions import Concat # noqa

import numpy as np
import argparse
import json


parser = argparse.ArgumentParser()
parser.add_argument("--modalities", default='[0,1,2]', type=str)
parser.add_argument("--dataset", default='mosi', type=str)
parser.add_argument("--dataset-path", default='/usr0/home/yuncheng/MultiBench/data/mosi_raw.pkl', type=str)
args = parser.parse_args()


# mosi_data.pkl, mosei_senti_data.pkl
# mosi_raw.pkl, mosei_raw.pkl, sarcasm.pkl, humor.pkl
# raw_path: mosi.hdf5, mosei.hdf5, sarcasm_raw_text.pkl, humor_raw_text.pkl
traindata, validdata, test_robust = \
    get_dataloader(args.dataset_path, robust_test=False, data_type=args.dataset)


modalities = json.loads(args.modalities)

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
    d_v = (371, 70)
    d_a = (81, 200)
    d_l = (300, 600)

config = [d_v, d_a, d_l]
d_modalities = [config[i] for i in modalities]
out_dim = sum([d[1] for d in d_modalities])
encoders = [Transformer(d[0], d[1]).cuda() for d in d_modalities]
head = MLP(out_dim, 256, 1).cuda()

fusion = Concat().cuda()

traintimes = []
mems = []
testtimes = []
accs = []
saved_model = '{}_models/{}_lf_transformer_{}.pt'.format(args.dataset, args.dataset, ''.join(list(map(str, modalities))))
for _ in range(5):
    traintime, mem, num_params = train(encoders, fusion, head, traindata, validdata, 50, task="regression", optimtype=torch.optim.AdamW,
       is_packed=True, lr=1e-4, save=saved_model, weight_decay=0.01, objective=torch.nn.L1Loss(), modalities=modalities)

    traintimes.append(traintime)
    mems.append(mem)

    print("Testing:")
    model = torch.load(saved_model).cuda()

    testtime, acc = test(model=model, test_dataloaders_all=test_robust, dataset=args.dataset, is_packed=True,
        criterion=torch.nn.L1Loss(), task='posneg-classification', no_robust=True, modalities=modalities)
    testtimes.append(testtime)
    accs.append(acc)

print("Average Training Time: {}, std: {}".format(np.mean(traintimes), np.std(traintimes)))
print("Average Training Memory: {}, std: {}".format(np.mean(mems), np.std(mems)))
print("Number of Parameters: {}".format(num_params))
print("Average Inference Time: {}, std: {}".format(np.mean(testtimes), np.std(testtimes)))
print("Average Performance: {}, std: {}".format(np.mean(accs), np.std(accs)))
