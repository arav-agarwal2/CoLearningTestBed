import torch
import torch.nn as nn
import sys
import os
sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))
from unimodals.common_models import MLP, Linear
from get_data import get_dataloader
from supervised_learning import train, test
from fusions.common_fusions import Concat
from cca_zoo.deepmodels import (
    DCCA,
    architectures,
    DCCA_NOI,
    DCCA_SDL,
    BarlowTwins,
)
import pytorch_lightning as pl
import torch
from torch.utils.data.dataloader import default_collate
from torchinfo import summary
import argparse
import pickle

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
parser.add_argument("--epochs", default=100, type=int)
parser.add_argument("--weight-decay", default=0.01, type=float)
parser.add_argument("--saved-model", default='/home/yuncheng/lrf_best.pt', type=str)
args = parser.parse_args()

with open(args.data_path, 'rb') as f:
  dicta = pickle.load(f)


train_tensor_list = [torch.tensor(dicta['train'][key]).float() for key in args.keys if key != 'label']
test_tensor_list = [torch.tensor(dicta['test'][key]).float() for key in args.keys if key != 'label']

train_dataset = torch.utils.data.TensorDataset(*train_tensor_list)
val_dataset = torch.utils.data.TensorDataset(*test_tensor_list)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.bs, collate_fn= lambda x: {"views": default_collate(x)})
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.bs, collate_fn= lambda x: {"views": default_collate(x)})
trainer = pl.Trainer(
    max_epochs=args.epochs,
    enable_checkpointing=False,
    log_every_n_steps=1,
    accelerator='cpu'
)

encoder_list = [architectures.Encoder(latent_dims=args.hidden_dim, feature_size=args.input_dim)]*len(args.modalities)
dcca = DCCA(latent_dims=args.hidden_dim, encoders=encoder_list)


trainer.fit(dcca, train_dataloader, val_dataloader)

# freeze parameters

for param in dcca.parameters():
    param.requires_grad = False


head = MLP((len(args.keys)-1)*args.hidden_dim, args.hidden_dim, args.num_classes).to(device)

fusion = Concat()




class DCCA_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = dcca
        self.head = head
        self.fusion = fusion
    def forward(self, inputs):
        encoded = self.encoder(inputs)
        fused = self.fusion(encoded)
        return self.head(fused)
        

model=  DCCA_model()

# Load data
traindata, validdata, testdata = get_dataloader(path=args.data_path, keys=args.keys, modalities=args.modalities, batch_size=args.bs, num_workers=args.num_workers)


optim = torch.optim.Adam(model.parameters(), lr=args.lr)
criterion = torch.nn.CrossEntropyLoss()
model.to('cpu')

for epoch in range(args.epochs):
    for *inputs, labels in traindata:
        #print(inputs, labels)
        optim.zero_grad()
        model_out = model([elem.float().to('cpu') for elem in inputs[0]])
        print(model_out.shape)
        loss = criterion(model_out, labels.to('cpu').reshape((-1,)))
        loss.backward()
        optim.step()

test(model, testdata, is_packed=False, no_robust=True, criterion=torch.nn.CrossEntropyLoss())
