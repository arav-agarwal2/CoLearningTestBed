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

class MFN(nn.Module):
	def __init__(self,config,NN1Config,NN2Config,gamma1Config,gamma2Config,outConfig):
		super(MFN, self).__init__()
		[self.d_l,self.d_a,self.d_v] = config["input_dims"]
		[self.dh_l,self.dh_a,self.dh_v] = config["h_dims"]
		total_h_dim = self.dh_l+self.dh_a+self.dh_v
		self.mem_dim = config["memsize"]
		window_dim = config["windowsize"]
		output_dim = 1
		attInShape = total_h_dim*window_dim
		gammaInShape = attInShape+self.mem_dim
		final_out = total_h_dim+self.mem_dim
		h_att1 = NN1Config["shapes"]
		h_att2 = NN2Config["shapes"]
		h_gamma1 = gamma1Config["shapes"]
		h_gamma2 = gamma2Config["shapes"]
		h_out = outConfig["shapes"]
		att1_dropout = NN1Config["drop"]
		att2_dropout = NN2Config["drop"]
		gamma1_dropout = gamma1Config["drop"]
		gamma2_dropout = gamma2Config["drop"]
		out_dropout = outConfig["drop"]

		self.lstm_l = nn.LSTMCell(self.d_l, self.dh_l)
		self.lstm_a = nn.LSTMCell(self.d_a, self.dh_a)
		self.lstm_v = nn.LSTMCell(self.d_v, self.dh_v)

		self.att1_fc1 = nn.Linear(attInShape, h_att1)
		self.att1_fc2 = nn.Linear(h_att1, attInShape)
		self.att1_dropout = nn.Dropout(att1_dropout)

		self.att2_fc1 = nn.Linear(attInShape, h_att2)
		self.att2_fc2 = nn.Linear(h_att2, self.mem_dim)
		self.att2_dropout = nn.Dropout(att2_dropout)

		self.gamma1_fc1 = nn.Linear(gammaInShape, h_gamma1)
		self.gamma1_fc2 = nn.Linear(h_gamma1, self.mem_dim)
		self.gamma1_dropout = nn.Dropout(gamma1_dropout)

		self.gamma2_fc1 = nn.Linear(gammaInShape, h_gamma2)
		self.gamma2_fc2 = nn.Linear(h_gamma2, self.mem_dim)
		self.gamma2_dropout = nn.Dropout(gamma2_dropout)

		self.out_fc1 = nn.Linear(final_out, h_out)
		self.out_fc2 = nn.Linear(h_out, output_dim)
		self.out_dropout = nn.Dropout(out_dropout)
		
	def forward(self,x):
		x_l = x[:,:,:self.d_l]
		x_a = x[:,:,self.d_l:self.d_l+self.d_a]
		x_v = x[:,:,self.d_l+self.d_a:]
		# x is t x n x d
		n = x.shape[1]
		t = x.shape[0]
		self.h_l = torch.zeros(n, self.dh_l).cuda()
		self.h_a = torch.zeros(n, self.dh_a).cuda()
		self.h_v = torch.zeros(n, self.dh_v).cuda()
		self.c_l = torch.zeros(n, self.dh_l).cuda()
		self.c_a = torch.zeros(n, self.dh_a).cuda()
		self.c_v = torch.zeros(n, self.dh_v).cuda()
		self.mem = torch.zeros(n, self.mem_dim).cuda()
		all_h_ls = []
		all_h_as = []
		all_h_vs = []
		all_c_ls = []
		all_c_as = []
		all_c_vs = []
		all_mems = []
		for i in range(t):
			# prev time step
			prev_c_l = self.c_l
			prev_c_a = self.c_a
			prev_c_v = self.c_v
			# curr time step
			new_h_l, new_c_l = self.lstm_l(x_l[i], (self.h_l, self.c_l))
			new_h_a, new_c_a = self.lstm_a(x_a[i], (self.h_a, self.c_a))
			new_h_v, new_c_v = self.lstm_v(x_v[i], (self.h_v, self.c_v))
			# concatenate
			prev_cs = torch.cat([prev_c_l,prev_c_a,prev_c_v], dim=1)
			new_cs = torch.cat([new_c_l,new_c_a,new_c_v], dim=1)
			cStar = torch.cat([prev_cs,new_cs], dim=1)
			attention = F.softmax(self.att1_fc2(self.att1_dropout(F.relu(self.att1_fc1(cStar)))),dim=1)
			attended = attention*cStar
			cHat = F.tanh(self.att2_fc2(self.att2_dropout(F.relu(self.att2_fc1(attended)))))
			both = torch.cat([attended,self.mem], dim=1)
			gamma1 = F.sigmoid(self.gamma1_fc2(self.gamma1_dropout(F.relu(self.gamma1_fc1(both)))))
			gamma2 = F.sigmoid(self.gamma2_fc2(self.gamma2_dropout(F.relu(self.gamma2_fc1(both)))))
			self.mem = gamma1*self.mem + gamma2*cHat
			all_mems.append(self.mem)
			# update
			self.h_l, self.c_l = new_h_l, new_c_l
			self.h_a, self.c_a = new_h_a, new_c_a
			self.h_v, self.c_v = new_h_v, new_c_v
			all_h_ls.append(self.h_l)
			all_h_as.append(self.h_a)
			all_h_vs.append(self.h_v)
			all_c_ls.append(self.c_l)
			all_c_as.append(self.c_a)
			all_c_vs.append(self.c_v)

		# last hidden layer last_hs is n x h
		last_h_l = all_h_ls[-1]
		last_h_a = all_h_as[-1]
		last_h_v = all_h_vs[-1]
		last_mem = all_mems[-1]
		last_hs = torch.cat([last_h_l,last_h_a,last_h_v,last_mem], dim=1)
		output = self.out_fc2(self.out_dropout(F.relu(self.out_fc1(last_hs))))
		return output

