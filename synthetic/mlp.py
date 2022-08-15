import torch
import sys
import os
sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))
from unimodals.common_models import MLP, Identity
from get_data import get_dataloader
from supervised_learning import train, test
from fusions.common_fusions import Concat

import argparse
import json


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument("--bs", default=32, type=int)
parser.add_argument("--num-workers", default=4, type=int)
parser.add_argument("--input-dim", default=30, type=int)
parser.add_argument("--hidden-dim", default=512, type=int)
parser.add_argument("--modalities", default='[0,1]', type=str)
parser.add_argument("--epochs", default=100, type=int)
parser.add_argument("--eval", default=True, type=int)
parser.add_argument("--output-path", default='/home/yuncheng/best.pt', type=str)
args = parser.parse_args()

traindata, validdata, testdata = get_dataloader(path="/home/yuncheng/MultiBench/synthetic/SIMPLE_DATA_CLASS=2_DIM=1_STD=0.5.pickle", batch_size=args.bs, num_workers=args.num_workers)

modalities = json.loads(args.modalities)
input_dim = args.input_dim * len(modalities)
encoders = [Identity().to(device), Identity().to(device)]
fusion = Concat().to(device)
head = MLP(input_dim, args.hidden_dim, 2).to(device)
train(encoders, fusion, head, traindata, testdata, args.epochs, optimtype=torch.optim.Adam, is_packed=False, lr=1e-4, save=args.output_path, weight_decay=0, objective=torch.nn.CrossEntropyLoss(), modalities=modalities)

print("Testing:")
model = torch.load(args.output_path).to(device)
test(model, testdata, is_packed=False, no_robust=True, criterion=torch.nn.CrossEntropyLoss(), modalities=modalities)
