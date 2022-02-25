import torch
import random
from torch import nn
from torch.autograd import Variable


# Simple concatenation on dim 1
class Concat(nn.Module):
    def forward(self, modalities):
        flattened = []
        for modality in modalities:
            flattened.append(torch.flatten(modality, start_dim=1))
        return torch.cat(flattened, dim=1)


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    # change teacher_forcing_ratio to 0.0 when evaluating
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.size(1)
        max_len = src.size(0)

        output_size = self.decoder.output_size
        outputs = Variable(torch.zeros(
            max_len, batch_size, output_size)).cuda()

        encoder_output, hidden = self.encoder(src)
        hidden = hidden[:self.decoder.n_layers]

        if self.training:
            output = Variable(
                torch.zeros_like(trg.data[0, :]))  # solve the bug of input.size must be equal to input_size
        else:
            output = Variable(torch.zeros_like(src.data[0, :]))
        for t in range(0, max_len):
            output, hidden, _ = self.decoder(
                output, hidden, encoder_output)
            outputs[t] = output

            is_teacher = random.random() < teacher_forcing_ratio
            if is_teacher:
                output = Variable(trg.data[t]).cuda()

        return outputs, encoder_output


class L2MCTN(nn.Module):
    def __init__(self, seq2seq_0, seq2seq_1, regression_encoder, head, p=0.2):
        super(L2MCTN, self).__init__()
        self.seq2seq0 = seq2seq_0
        self.seq2seq1 = seq2seq_1

        self.dropout = nn.Dropout(p)
        self.regression = regression_encoder
        self.head = head

    def forward(self, src, trg0=None, trg1=None):
        reout = None
        rereout = None
        if self.training:
            out, _ = self.seq2seq0(src, trg0)
            reout, joint_embbed0 = self.seq2seq0(out, src)
            rereout, joint_embbed1 = self.seq2seq1(joint_embbed0, trg1)
        else:
            out, _ = self.seq2seq0(src, trg0, teacher_forcing_ratio=0.0)
            _, joint_embbed0 = self.seq2seq0.encoder(out)
            _, joint_embbed1 = self.seq2seq1.encoder(joint_embbed0)
        _, reg = self.regression(joint_embbed1)
        reg = self.dropout(reg)
        head_out = self.head(reg)[0]
        head_out = self.dropout(head_out)
        return out, reout, rereout, head_out
