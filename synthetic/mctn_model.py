import torch
from torch import nn
from torch.nn import functional as F


class MLP(nn.Module):

    def __init__(self, indim, hiddim, outdim, dropout=False, dropoutp=0.1):
        super(MLP, self).__init__()
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
        return self.lklu(output2)


class Translation(nn.Module):
    
    def __init__(self, encoder, decoder):
        super(Translation, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src):
        joint_embbed = self.encoder(src)
        out = self.decoder(joint_embbed)
        return out, joint_embbed


class MCTN(nn.Module):

    def __init__(self, translations, head, p=0.2):
        super(MCTN, self).__init__()
        self.translations = translations
        self.dropout = nn.Dropout(p)
        self.head = head

    def forward(self, src, trgs):
        out, _ = self.translations[0](src)
        outs = [out]
        reouts = None
        if self.training:
            reout, joint_embbed = self.translations[0](out)
            reouts = [reout]
            for i in range(1, len(trgs)-1):
                out, _ = self.translations[i](joint_embbed)
                reout, joint_embbed = self.translations[i](out)
                outs.append(out)
                reouts.append(reout)
            out, joint_embbed = self.translations[-1](joint_embbed)
            outs.append(out)
        else:
            for i in range(1, len(trgs)):
                joint_embbed = self.translations[i-1].encoder(out)
                out, _ = self.translations[i](joint_embbed)
                outs.append(out)
            joint_embbed = self.translations[-1].encoder(out)
        head_out = self.head(joint_embbed)
        head_out = self.dropout(head_out)
        return outs, reouts, head_out
