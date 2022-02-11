import torch
import torchvision

from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class MLP(torch.nn.Module):
    """
    Two layer MLP
    """

    def __init__(self, indim, hiddim, outdim, dropout=False, dropoutp=0.1, output_each_layer=False):
        super(MLP, self).__init__()
        self.fc = nn.Linear(indim, hiddim)
        self.fc2 = nn.Linear(hiddim, outdim)
        self.dropout_layer = torch.nn.Dropout(dropoutp)
        self.dropout = dropout
        self.output_each_layer = output_each_layer
        self.lklu = nn.LeakyReLU(0.2)

    def forward(self, x):
        output = F.relu(self.fc(x))
        if self.dropout:
            output = self.dropout_layer(output)
        output2 = self.fc2(output)
        if self.dropout:
            output2 = self.dropout_layer(output)
        if self.output_each_layer:
            return [0, x, output, self.lklu(output2)]
        return output2


class GRU(torch.nn.Module):
    """
    GRU Implementation.
    """

    def __init__(self, indim, hiddim, dropout=False, dropoutp=0.1, flatten=False, has_padding=False, last_only=False, batch_first=True):
        super(GRU, self).__init__()
        self.gru = nn.GRU(indim, hiddim, batch_first=True)
        self.dropout = dropout
        self.dropout_layer = torch.nn.Dropout(dropoutp)
        self.flatten = flatten
        self.has_padding = has_padding
        self.last_only = last_only
        self.batch_first = batch_first

    def forward(self, x):

        if self.has_padding:
            x = pack_padded_sequence(
                x[0], x[1], batch_first=self.batch_first, enforce_sorted=False)
            out = self.gru(x)[1][-1]
        elif self.last_only:
            out = self.gru(x)[1][0]

            return out
        else:
            out, l = self.gru(x)
            print(l.size())
        if self.dropout:
            out = self.dropout_layer(out)
        if self.flatten:
            out = torch.flatten(out, 1)

        return out
