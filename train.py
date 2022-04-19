"""Training scripts"""
import random
import numpy as np
import torch
from torch import nn
from eval_metrics import accuracy


softmax = nn.Softmax()


class MMDL(nn.Module):
    def __init__(self, encoders, fusion, head, has_padding=False):
        super(MMDL, self).__init__()
        self.encoders = nn.ModuleList(encoders)
        self.fuse = fusion
        self.head = head
        self.has_padding = has_padding
        self.fuseout = None
        self.reps = []

    def forward(self, inputs):
        outs = []
        if self.has_padding:
            for i in range(len(inputs[0])):
                outs.append(self.encoders[i](
                    [inputs[0][i], inputs[1][i]]))
        else:
            for i in range(len(inputs)):
                outs.append(self.encoders[i](inputs[i]))
        self.reps = outs
        if self.has_padding:

            if isinstance(outs[0], torch.Tensor):
                out = self.fuse(outs)
            else:
                out = self.fuse([i[0] for i in outs])
        else:
            out = self.fuse(outs)
        self.fuseout = out
        if type(out) is tuple:
            out = out[0]
        if self.has_padding and not isinstance(outs[0], torch.Tensor):
            return self.head([out, inputs[1][0]])
        return self.head(out)


def deal_with_objective(objective, pred, truth, args):
    if type(objective) == nn.CrossEntropyLoss:
        if len(truth.size()) == len(pred.size()):
            truth1 = truth.squeeze(len(pred.size())-1)
        else:
            truth1 = truth
        return objective(pred, truth1.long().cuda())
    elif type(objective) == nn.MSELoss or type(objective) == nn.modules.loss.BCEWithLogitsLoss or type(
            objective) == nn.L1Loss:
        return objective(pred, truth.float().cuda())
    else:
        return objective(pred, truth, args)


def train(encoders, fusion, head, train_dataloader, valid_dataloader, total_epochs,
          additional_optimizing_modules=[], is_packed=False, early_stop=False, task="classification",
          optimtype=torch.optim.RMSprop, lr=0.001, weight_decay=0.0, objective=nn.CrossEntropyLoss(),
          save='best.pt', objective_args_dict=None, input_to_float=True, clip_val=8):
    model = MMDL(encoders, fusion, head, has_padding=is_packed).cuda()

    def trainprocess():
        additional_params = []
        for m in additional_optimizing_modules:
            additional_params.extend(
                [p for p in m.parameters() if p.requires_grad])
        op = optimtype([p for p in model.parameters() if p.requires_grad] +
                       additional_params, lr=lr, weight_decay=weight_decay)
        bestvalloss = 10000
        bestacc = 0
        patience = 0

        def processinput(inp):
            if input_to_float:
                return inp.float()
            else:
                return inp

        for epoch in range(total_epochs):
            totalloss = 0.0
            totals = 0
            model.train()
            for j in train_dataloader:
                op.zero_grad()
                if is_packed:
                    with torch.backends.cudnn.flags(enabled=False):
                        model.train()
                        out = model([[processinput(i).cuda()
                                      for i in j[0]], j[1]])

                else:
                    model.train()
                    out = model([processinput(i).cuda()
                                 for i in j[:-1]])
                if not (objective_args_dict is None):
                    objective_args_dict['reps'] = model.reps
                    objective_args_dict['fused'] = model.fuseout
                    objective_args_dict['inputs'] = j[:-1]
                    objective_args_dict['training'] = True
                    objective_args_dict['model'] = model
                loss = deal_with_objective(
                    objective, out, j[-1], objective_args_dict)

                totalloss += loss * len(j[-1])
                totals += len(j[-1])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_val)
                op.step()
            print("Epoch "+str(epoch)+" train loss: "+str(totalloss/totals))
            model.eval()
            with torch.no_grad():
                totalloss = 0.0
                pred = []
                true = []
                for j in valid_dataloader:
                    if is_packed:
                        model.train()
                        out = model([[processinput(i).cuda()
                                      for i in j[0]], j[1]])
                    else:
                        model.train()
                        out = model([processinput(i).cuda()
                                     for i in j[:-1]])

                    if not (objective_args_dict is None):
                        objective_args_dict['reps'] = model.reps
                        objective_args_dict['fused'] = model.fuseout
                        objective_args_dict['inputs'] = j[:-1]
                        objective_args_dict['training'] = False
                    loss = deal_with_objective(
                        objective, out, j[-1], objective_args_dict)
                    totalloss += loss*len(j[-1])
                    if task == "classification":
                        pred.append(torch.argmax(out, 1))
                    true.append(j[-1])
            if pred:
                pred = torch.cat(pred, 0)
            true = torch.cat(true, 0)
            totals = true.shape[0]
            valloss = totalloss/totals
            if task == "classification":
                acc = accuracy(true, pred)
                print("Epoch "+str(epoch)+" valid loss: "+str(valloss) + " acc: "+str(acc))
                if acc > bestacc:
                    patience = 0
                    bestacc = acc
                    print("Saving Best")
                    torch.save(model, save)
            elif task == "regression":
                print("Epoch "+str(epoch)+" valid loss: "+str(valloss.item()))
                if valloss < bestvalloss:
                    patience = 0
                    bestvalloss = valloss
                    print("Saving Best")
                    torch.save(model, save)
                else:
                    patience += 1
            else:
                patience += 1
            if early_stop and patience > 7:
                break
    trainprocess()

def train_mfn(model, config, train_dataloader, valid_dataloader, criterion=nn.L1Loss(), optimtype=torch.optim.Adam, model_save='best_mfn.pt'):
    [config,NN1Config,NN2Config,gamma1Config,gamma2Config,outConfig] = configs
    model = model.cuda()
    op = optimtype(model.parameters(), lr=config["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(op,mode='min',patience=100,factor=0.5,verbose=True)

    def trainprocess():
        total_epochs = config['num_epochs']
        bestacc = 0
        patience = 0
        for epoch in range(total_epochs):
            totalloss = 0.0
            totals = 0
            model.train()
            for j in train_dataloader:
                op.zero_grad()
                predictions = model.forward(j[:-1]).squeeze(1)
                loss = criterion(predictions, j[-1])
                totalloss += loss * len(j[-1])
                totals += len(j[-1])
                loss.backward()
                op.step()
            print("Epoch "+str(epoch)+" train loss: "+str(totalloss/totals))
            model.eval()
            with torch.no_grad():
                totalloss = 0.0
                pred = []
                true = []
                for j in valid_dataloader:
                    predictions = model.forward(j[:-1]).squeeze(1)
                    pred.append[predictions]
                    loss = criterion(predictions, j[-1])
                    totalloss += loss * len(j[-1])
                    true.append(j[-1])
                pred = torch.cat(pred, 0)
                true = torch.cat(true, 0)
                totals = true.shape[0]
                valloss = totalloss/totals
                acc = accuracy(true, pred)
                print("Epoch "+str(epoch)+" valid loss: "+str(valloss) + " acc: "+str(acc))
                if acc > bestacc:
                    patience = 0
                    bestacc = acc
                    print("Saving Best")
                    torch.save(model, model_save)
                else:
                    patience += 1
                if patience > 7:
                    break
    trainprocess()
