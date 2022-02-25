from torch import nn
import torch
import sys
import os

sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))

from train import train
from test import test
# from training_structures.MCTN_Level2 import train, test  # noqa
from encoders import MLP, Encoder, Decoder 
from mosi_get_data import get_dataloader


# mosi_data.pkl, mosei_senti_data.pkl
# mosi_raw.pkl, mosei_raw.pkl, sarcasm.pkl, humor.pkl
# raw_path: mosi.hdf5, mosei.hdf5, sarcasm_raw_text.pkl, humor_raw_text.pkl
traindata, validdata, testdata = get_dataloader('/content/mosi_raw.pkl')

max_seq = 20
feature_dim = 300
hidden_dim = 32

encoder0 = Encoder(feature_dim, hidden_dim, n_layers=1, dropout=0.0).cuda()
decoder0 = Decoder(hidden_dim, feature_dim, n_layers=1, dropout=0.0).cuda()
encoder1 = Encoder(hidden_dim, hidden_dim, n_layers=1, dropout=0.0).cuda()
decoder1 = Decoder(hidden_dim, feature_dim, n_layers=1, dropout=0.0).cuda()

reg_encoder = nn.GRU(hidden_dim, 32).cuda()
head = MLP(32, 64, 1).cuda()


train(
        traindata, validdata,
        encoder0, decoder0, encoder1, decoder1,
        reg_encoder, head,
        criterion_t0=nn.MSELoss(), criterion_c=nn.MSELoss(),
        criterion_t1=nn.MSELoss(), criterion_r=nn.L1Loss(),
        max_seq_len=20,
        mu_t0=0.01, mu_c=0.01, mu_t1=0.01,
        dropout_p=0.15, early_stop=False, patience_num=15,
        lr=1e-4, weight_decay=0.01, op_type=torch.optim.AdamW,
        epoch=200, model_save='best_mctn.pt')

print("Testing:")
model = torch.load('best_mctn.pt').cuda()

test(model, testdata, 'mosi')
