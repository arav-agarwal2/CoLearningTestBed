"""Training scripts"""
import random
import numpy as np
import torch
from torch import nn
from eval_metrics import accuracy


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train(model, configs, train_dataloader, valid_dataloader, criterion=nn.L1Loss(), optimtype=torch.optim.Adam, model_save='best_mfn.pt'):
    [config,NN1Config,NN2Config,gamma1Config,gamma2Config,outConfig] = configs
    model = model.cuda()
    op = optimtype(model.parameters(), lr=config["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(op,mode='min',patience=100,factor=0.5,verbose=True)

    def trainprocess():
        total_epochs = config['num_epochs']
        best_loss = 999999.0
        patience = 0
        for epoch in range(total_epochs):
            totalloss = 0.0
            totals = 0
            model.train()
            for j in train_dataloader:
                op.zero_grad()
                predictions = model.forward(j[0]).squeeze(1)
                label = j[-1].to(device).squeeze()
                loss = criterion(predictions, label)
                totalloss += loss * len(label)
                totals += len(label)
                loss.backward()
                op.step()
            print("Epoch "+str(epoch)+" train loss: "+str(totalloss/totals))
            model.eval()
            with torch.no_grad():
                totalloss = 0.0
                pred = []
                true = []
                for j in valid_dataloader:
                    predictions = model.forward(j[0]).squeeze(1)
                    pred.append(predictions)
                    label = j[-1].to(device).squeeze()
                    loss = criterion(predictions, label)
                    totalloss += loss * len(label)
                    true.append(label)
                pred = torch.cat(pred, 0)
                true = torch.cat(true, 0)
                totals = true.shape[0]
                valloss = totalloss/totals
                print("Epoch "+str(epoch)+" valid loss: "+str(valloss))
                if valloss < best_loss:
                    patience = 0
                    best_loss = valloss
                    print("Saving Best")
                    torch.save(model, model_save)
                else:
                    patience += 1
                if patience > 7:
                    break
    trainprocess()
