"""Training scripts"""
import random
import numpy as np
import torch
from torch import nn
from sklearn.metrics import accuracy_score, f1_score


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

def test(model, test_dataloader):
    with torch.no_grad():
        pred = []
        true = []
        for j in test_dataloader:
            model.eval()
            predictions = model.forward(j[0]).cpu().data.numpy()
            pred.append(predictions)
            label = j[-1]
            true.append(label)
        pred = np.concatenate(pred, 0)
        true = np.concatenate(true, 0)
        mae = np.mean(np.absolute(pred - true))
        mult = round(np.sum(np.round(pred) == np.round(true))/float(len(true)), 5)
        f_score = round(f1_score(np.round(pred),np.round(true), average='weighted'), 5)
        true_label = (true >= 0)
        predicted_label = (pred >= 0)
        acc = accuracy_score(true_label, predicted_label)
        print('Test: MAE: {}, Mult Acc: {}, Mult f_score: {}, Acc: {}'.format(mae, mult, f_score, acc))
