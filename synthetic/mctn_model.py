import torch
from torch import nn
from torch.nn import functional as F


class MLP(nn.Module):

    def __init__(self, indim, hiddim, outdim, dropout=False, dropoutp=0.1):
        super(MLP, self).__init__()
        self.indim = indim
        self.outdim = outdim
        self.fc = nn.Linear(indim, hiddim)
        self.fc2 = nn.Linear(hiddim, outdim)
        self.dropout_layer = torch.nn.Dropout(dropoutp)
        self.dropout = dropout
        self.lklu = nn.LeakyReLU(0.2)

    def forward(self, x):
        output = F.relu(self.fc(x))
        if self.dropout:
            output = self.dropout_layer(output)
        output2 = self.fc2(output)
        if self.dropout:
            output2 = self.dropout_layer(output)
        return output2


class Translation(nn.Module):
    
    def __init__(self, encoder, decoder, i, dropoutp=0.1):
        super(Translation, self).__init__()
        if i != 0:
            self.input = nn.Linear(encoder.outdim, encoder.indim)
            self.dropout_layer = torch.nn.Dropout(dropoutp)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src):
        # if src.shape[1] != self.encoder.indim:
        #     src = self.input(src)
        #     src = self.dropout_layer(src)
        joint_embbed = self.encoder(src)
        out = self.decoder(joint_embbed)
        return out, joint_embbed


class MCTN(nn.Module):

    def __init__(self, translations, head, p=0.2):
        super(MCTN, self).__init__()
        self.translations = nn.ModuleList(translations)
        self.dropout = nn.Dropout(p)
        self.head = head

    def forward(self, src, trgs):
        out, joint_embbed = self.translations[0](src)
        outs = [out]
        reouts = []
        if self.training:
            # reout, joint_embbed = self.translations[0](out)
            # reouts = [reout]
            for i in range(1, len(trgs)-1):
                out, joint_embbed = self.translations[i](joint_embbed)
                # reout, joint_embbed = self.translations[i](out)
                outs.append(out)
                # reouts.append(reout)
            out, joint_embbed = self.translations[-1](joint_embbed)
            outs.append(out)
        else:
            for i in range(1, len(trgs)):
                # joint_embbed = self.translations[i-1].encoder(out)
                out, joint_embbed = self.translations[i](joint_embbed)
                outs.append(out)
            # joint_embbed = self.translations[-1].encoder(out)
        head_out = self.head(joint_embbed)
        head_out = self.dropout(head_out)
        return outs, reouts, head_out