import torch
import sys
import os
sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))
from mctn_model import MLP
from get_data import get_dataloader
from mctn_learning import train, test

import argparse


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument("--data-path", default="/home/yuncheng/MultiBench/synthetic/SIMPLE_DATA_DIM=3_STD=0.5.pickle", type=str, help="input path of synthetic dataset")
parser.add_argument("--keys", nargs='+', default=['a','b','c','d','e','label'], type=list, help="keys to access data of each modality and label, assuming dataset is structured as a dict")
parser.add_argument("--modalities", nargs='+', default=[0,1], type=int, help="specify the index of modalities in keys")
parser.add_argument("--bs", default=128, type=int)
parser.add_argument("--num-workers", default=4, type=int)
parser.add_argument("--input-dim", default=30, type=int)
parser.add_argument("--hidden-dim", default=512, type=int)
parser.add_argument("--num-classes", default=2, type=int)
parser.add_argument("--epochs", default=100, type=int)
parser.add_argument("--lr", default=1e-3, type=float)
parser.add_argument("--weight-decay", default=0.01, type=float)
parser.add_argument("--saved-model", default=None, type=str)
args = parser.parse_args()
torch.multiprocessing.set_sharing_strategy('file_system')

# Load data
traindata, _, testdata = get_dataloader(path=args.data_path, keys=args.keys, modalities=args.modalities, batch_size=args.bs, num_workers=args.num_workers)

# Specify MCTN model
encoders = [MLP(args.input_dim, args.hidden_dim//2, args.hidden_dim).to(device)] + [MLP(args.hidden_dim, args.hidden_dim//2, args.hidden_dim).to(device)] * (len(args.modalities)-1)
decoders = [MLP(args.hidden_dim, args.hidden_dim//2, args.input_dim).to(device) for _ in args.modalities]
head = MLP(args.hidden_dim, args.hidden_dim*2, args.num_classes).to(device)

# Train the translation first
train(traindata, testdata, encoders, decoders, head, epoch=args.epochs, level=len(args.modalities), op_type=torch.optim.AdamW, lr=args.lr, save=args.saved_model, weight_decay=args.weight_decay)
# model = torch.load(args.saved_model).to(device)
# for translation in model.translations:
#     for name, param in translation.named_parameters():
#         param.requires_grad = False
# # Then train classification
# train(traindata, testdata, encoders, decoders, head, epoch=args.epochs, model=model, level=len(args.modalities), op_type=torch.optim.AdamW, lr=args.lr, save=args.saved_model, weight_decay=args.weight_decay)

# Testing
# print("Testing:")
# model = torch.load(args.saved_model).to(device)
# test(model, testdata)
