import torch
from torch import nn
from torch.autograd import Variable

from log_uniform.log_uniform import LogUniformSampler


def get_rnn(rnn_type):
    return getattr(nn, rnn_type)


def init_lstm(lstm, wb=None, one_f=True):
    for i in range(lstm.num_layers):
        if wb:
            getattr(lstm, 'weight_ih_l%d' % i).data.uniform_(-wb, wb)
            getattr(lstm, 'weight_hh_l%d' % i).data.uniform_(-wb, wb)
        if one_f:
            nhid = lstm.hidden_size
            getattr(lstm, 'bias_ih_l%d' % i)[nhid:2 * nhid].data.fill_(1.0)
            getattr(lstm, 'bias_hh_l%d' % i)[nhid:2 * nhid].data.fill_(1.0)


class WordRNN(nn.Module):
    def __init__(self, rnn_type, ntokens, nlayers, nhid, din, dout,
                 dropout=0.5, pad_idx=-1):
        super(WordRNN, self).__init__()
        self.drop = nn.Dropout(dropout)

        self.encoder = nn.Embedding(ntokens, din, padding_idx=pad_idx)

        self.rnn_type = rnn_type.upper()
        self.rnn = get_rnn(self.rnn_type)(din, nhid, nlayers, dropout=dropout)

        self.fc = nn.Linear(nhid, dout, bias=False)
        self.decoder = nn.Linear(dout, ntokens, bias=False)

        self.init_weights(0.05)

        self.ntokens = ntokens

    def init_weights(self, initrange):
        for p in self.parameters():
            p.data.uniform_(-initrange, initrange)
        if self.rnn_type == 'LSTM':
            init_lstm(self.rnn, wb=None, one_f=True)

    def forward(self, input, hidden=None, decode=True):
        emb = self.encoder(input)
        output, hidden = self.rnn(emb, hidden)
        output = self.fc(output)

        if decode:
            decoded = self.decoder(output)
            return decoded, hidden
        else:
            return output, hidden


class WordbiRNN(nn.Module):
    def __init__(self, rnn_type, ntokens, nlayers, nhid, din, dout,
                 dropout=0.5, pad_idx=-1, tie_bidirec=False):
        super(WordbiRNN, self).__init__()
        self.drop = nn.Dropout(dropout)

        self.fw_lookup_table = nn.Embedding(ntokens, din, padding_idx=pad_idx)
        self.bw_lookup_table = (self.fw_lookup_table if tie_bidirec else
                                nn.Embedding(ntokens, din, padding_idx=pad_idx))

        self.rnn_type = rnn_type.upper()
        self.fw_rnn = get_rnn(self.rnn_type)(din, nhid, nlayers, dropout=dropout)
        self.bw_rnn = get_rnn(self.rnn_type)(din, nhid, nlayers, dropout=dropout)

        self.fw_fc = nn.Linear(nhid, dout, bias=False)
        self.bw_fc = nn.Linear(nhid, dout, bias=False)
        self.decoder = nn.Linear(dout, ntokens, bias=False)

        self.init_weights(0.05)

        self.ntokens = ntokens
        self.tie_bidirec = tie_bidirec

    def init_weights(self, initrange):
        for p in self.parameters():
            p.data.uniform_(-initrange, initrange)
        if self.rnn_type == 'LSTM':
            init_lstm(self.fw_rnn, wb=None, one_f=True)
            init_lstm(self.bw_rnn, wb=None, one_f=True)

    def forward(self, input, hiddens=None, decode=True):
        inv_idx = torch.arange(input.size(0) - 1, -1, -1, out=input.data.new())

        fw_emb = self.fw_lookup_table(input)
        bw_emb = self.bw_lookup_table(input[inv_idx])

        if hiddens is None:
            fw_hidden, bw_hidden = None, None
        else:
            fw_hidden, bw_hidden = hiddens

        fw_output, fw_hidden = self.fw_rnn(fw_emb, fw_hidden)
        bw_output, bw_hidden = self.bw_rnn(bw_emb, bw_hidden)

        fw_out = fw_output[:-2]
        bw_out = bw_output[inv_idx][2:]

        fw_fc_out = self.fw_fc(self.drop(fw_out))
        bw_fc_out = self.bw_fc(self.drop(bw_out))

        fc_output = 0.5 * (fw_fc_out + bw_fc_out)
        hiddens = (fw_hidden, bw_hidden)

        if decode:
            decoded = self.decoder(fc_output)
            return decoded, hiddens
        else:
            return fc_output, hiddens


class SampledSoftmax(nn.Module):
    def __init__(self, ntokens, nsampled, decoding_module):
        super(SampledSoftmax, self).__init__()

        # Parameters
        self.ntokens = ntokens
        self.nsampled = nsampled

        self.sampler = LogUniformSampler(self.ntokens)
        self.params = decoding_module

    def forward(self, inputs, labels):
        if self.training:
            return self.sampled(inputs, labels, remove_accidental_match=True)
        else:
            return self.full(inputs, labels)

    def sampled(self, inputs, labels, remove_accidental_match=False):
        batch_size, d = inputs.size()
        labels_ndarray = labels.data.cpu().numpy()
        sample_ids, true_freq, sample_freq = self.sampler.sample(self.nsampled, labels_ndarray)

        # gather true labels and sample ids
        true_weights = self.params.weight.index_select(0, labels)
        sample_weights = self.params.weight[sample_ids, :]

        # calculate logits
        true_logits = torch.sum(torch.mul(inputs, true_weights), dim=1)
        sample_logits = torch.matmul(inputs, torch.t(sample_weights))
        # remove true labels from sample set
        if remove_accidental_match:
            acc_hits = self.sampler.accidental_match(labels_ndarray, sample_ids)
            if len(acc_hits) > 0:
                acc_hits = list(zip(*acc_hits))
                sample_logits[acc_hits] = -1e37

        # perform correction
        true_freq = Variable(type(inputs.data)(true_freq))
        sample_freq = Variable(type(inputs.data)(sample_freq))

        true_logits = true_logits.sub(torch.log(true_freq))
        sample_logits = sample_logits.sub(torch.log(sample_freq))

        # return logits and new_labels
        logits = torch.cat((torch.unsqueeze(true_logits, dim=1), sample_logits), dim=1)
        new_targets = Variable(type(labels.data)(batch_size).zero_())
        return logits, new_targets

    def full(self, inputs, labels):
        return self.params(inputs), labels
