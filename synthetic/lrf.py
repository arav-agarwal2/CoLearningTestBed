import torch
import sys
import os
sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))
from unimodals.common_models import MLP, Linear
from get_data import get_dataloader
from supervised_learning import train, test
from fusions.common_fusions import LowRankTensorFusion

import argparse


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument("--data-path", default="/home/yuncheng/MultiBench/synthetic/SIMPLE_DATA_DIM=3_STD=0.5.pickle", type=str, help="input path of synthetic dataset")
parser.add_argument("--keys", nargs='+', default=['a','b','c','d','e','label'], type=list, help="keys to access data of each modality and label, assuming dataset is structured as a dict")
parser.add_argument("--modalities", nargs='+', default=[0,1], type=int, help="specify the index of modalities in keys")
parser.add_argument("--bs", default=32, type=int)
parser.add_argument("--num-workers", default=4, type=int)
parser.add_argument("--input-dim", default=30, type=int)
parser.add_argument("--output-dim", default=128, type=int)
parser.add_argument("--hidden-dim", default=512, type=int)
parser.add_argument("--rank", default=32, type=int)
parser.add_argument("--num-classes", default=2, type=int)
parser.add_argument("--epochs", default=200, type=int)
parser.add_argument("--lr", default=3e-4, type=float)
parser.add_argument("--weight-decay", default=0.01, type=float)
parser.add_argument("--saved-model", default='/home/yuncheng/lrf_best.pt', type=str)
args = parser.parse_args()

# Load data
traindata, validdata, testdata = get_dataloader(path=args.data_path, keys=args.keys, modalities=args.modalities, batch_size=args.bs, num_workers=args.num_workers)

# Specify late rank fusion model
in_dim = [args.output_dim] * len(args.modalities)
encoders = [Linear(args.input_dim, args.output_dim).to(device) for _ in args.modalities]
head = MLP(args.output_dim, args.hidden_dim, args.num_classes).to(device)
fusion = LowRankTensorFusion(in_dim, args.output_dim, args.rank).to(device)

# Training
train(encoders, fusion, head, traindata, testdata, args.epochs, optimtype=torch.optim.AdamW, is_packed=False, lr=args.lr, save=args.saved_model, weight_decay=args.weight_decay, objective=torch.nn.CrossEntropyLoss())

# Testing
print("Testing:")
model = torch.load(args.saved_model).to(device)
test(model, testdata, is_packed=False, no_robust=True, criterion=torch.nn.CrossEntropyLoss())
