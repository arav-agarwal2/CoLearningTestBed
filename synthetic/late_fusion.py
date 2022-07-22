import torch
import sys
import os
sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))
from unimodals.common_models import MLP, Linear
from get_data import get_dataloader
from supervised_learning import train, test
from fusions.common_fusions import Concat

import argparse
import json
import ast

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument("--data-path", default="/home/yuncheng/MultiBench/synthetic/data.pickle", type=str, help="input path of synthetic dataset")
parser.add_argument("--keys", default="['a','b','label']", type=str, help="keys to access data of each modality and label, assuming dataset is structured as a dict")
parser.add_argument("--modalities", default='[0,1]', type=str, help="specify the index of modalities in keys")
parser.add_argument("--bs", default=32, type=int)
parser.add_argument("--num-workers", default=4, type=int)
parser.add_argument("--input-dim", default=30, type=int)
parser.add_argument("--output-dim", default=128, type=int)
parser.add_argument("--hidden-dim", default=512, type=int)
parser.add_argument("--num-classes", default=2, type=int)
parser.add_argument("--epochs", default=100, type=int)
parser.add_argument("--lr", default=1e-5, type=float)
parser.add_argument("--weight-decay", default=0, type=float)
parser.add_argument("--eval", default=True, type=int)
parser.add_argument("--saved-model", default='/home/yuncheng/late_fusion_best.pt', type=str)
args = parser.parse_args()

keys = ast.literal_eval(args.keys)
modalities = json.loads(args.modalities)

# Load data
traindata, validdata, testdata = get_dataloader(path=args.data_path, keys=keys, modalities=modalities, batch_size=args.bs, num_workers=args.num_workers)

# Specify late fusion model
out_dim = args.output_dim * len(modalities)
encoders = [Linear(args.input_dim, args.output_dim).to(device) for _ in modalities]
head = MLP(out_dim, args.hidden_dim, args.num_classes).to(device)
fusion = Concat().cuda()

# Training
train(encoders, fusion, head, traindata, validdata, args.epochs, optimtype=torch.optim.AdamW, early_stop=False, is_packed=False, lr=args.lr, save=args.saved_model, weight_decay=args.weight_decay, objective=torch.nn.CrossEntropyLoss(), modalities=modalities)

# Testing
print("Testing:")
model = torch.load(args.saved_model).to(device)
test(model, testdata, no_robust=True, criterion=torch.nn.CrossEntropyLoss(), modalities=modalities)
