from train import train
from test import test
from encoders import GRU, MLP
from mosi_get_data import get_dataloader
from fusion import Concat
import torch

traindata, validdata, testdata = get_dataloader(
    '/content/mosi_raw.pkl')

encoders = [GRU(35, 70, dropout=True, has_padding=True, batch_first=True).cuda(),
            GRU(74, 200, dropout=True, has_padding=True, batch_first=True).cuda(),
            GRU(300, 600, dropout=True, has_padding=True, batch_first=True).cuda()]
head = MLP(870, 870, 1).cuda()

fusion = Concat().cuda()

train(encoders, fusion, head, traindata, validdata, 100, task="regression", optimtype=torch.optim.AdamW,
      early_stop=False, is_packed=True, lr=1e-3, save='mosi_lf_best.pt', weight_decay=0.01, objective=torch.nn.L1Loss())

print("Testing:")
model = torch.load('mosi_lf_best.pt').cuda()

test(model=model, test_dataloader=testdata, is_packed=True, criterion=torch.nn.L1Loss(), task='posneg-classification')
