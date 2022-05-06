"""Memory Fusion Network"""
import random
import torch
from torch import nn
from fusion import MFN
from train_mfn import train, test
from mosi_get_data import get_dataloader

traindata, validdata, testdata = get_dataloader('/content/drive/MyDrive/colearning/mosi_raw.pkl')

config = dict()
config["input_dims"] = [300,74,35]
hl = random.choice([32,64,88,128,156,256])
ha = random.choice([8,16,32,48,64,80])
hv = random.choice([8,16,32,48,64,80])
config["h_dims"] = [hl,ha,hv]
config["memsize"] = random.choice([64,128,256,300,400])
config["windowsize"] = 2
config["batchsize"] = random.choice([32,64,128,256])
config["num_epochs"] = 50
config["lr"] = random.choice([0.001,0.002,0.005,0.008,0.01])
config["momentum"] = random.choice([0.1,0.3,0.5,0.6,0.8,0.9])
NN1Config = dict()
NN1Config["shapes"] = random.choice([32,64,128,256])
NN1Config["drop"] = random.choice([0.0,0.2,0.5,0.7])
NN2Config = dict()
NN2Config["shapes"] = random.choice([32,64,128,256])
NN2Config["drop"] = random.choice([0.0,0.2,0.5,0.7])
gamma1Config = dict()
gamma1Config["shapes"] = random.choice([32,64,128,256])
gamma1Config["drop"] = random.choice([0.0,0.2,0.5,0.7])
gamma2Config = dict()
gamma2Config["shapes"] = random.choice([32,64,128,256])
gamma2Config["drop"] = random.choice([0.0,0.2,0.5,0.7])
outConfig = dict()
outConfig["shapes"] = random.choice([32,64,128,256])
outConfig["drop"] = random.choice([0.0,0.2,0.5,0.7])
configs = [config, NN1Config, NN2Config, gamma1Config, gamma2Config, outConfig]
model = MFN(config, NN1Config, NN2Config, gamma1Config, gamma2Config, outConfig)

train(model, configs,
      traindata, validdata,
      criterion=nn.L1Loss(),
      optimtype=torch.optim.Adam,
      model_save='best_mfn.pt')

print("Testing:")
model = torch.load('best_mfn.pt').cuda()

test(model, testdata)
