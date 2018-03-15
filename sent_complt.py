import os
import glob
import re
import pickle

import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.autograd import Variable

from config import args, save_dir_root, sc_dir
from input import tokenize, Vocabulary


class SentenceCompletion(object):
    def __init__(self, prob, ans, sep=',', n_choices=5):
        self.prob_df = pd.read_table(os.path.join(sc_dir, prob),
                                     sep=sep, keep_default_na=False)
        self.ans_df = pd.read_table(os.path.join(sc_dir, ans), sep=sep)

        self.df = pd.merge(self.prob_df.ix[:, :n_choices + 2],
                           self.ans_df.ix[:, :2], left_on='id', right_on='id')
        self.choice_names = ['C%d' % i for i in range(1, n_choices + 1)]
        self.df.columns = ['No', 'Q'] + self.choice_names + ['A']
        self.df['Q'] = self.df['Q'].replace('.', ' . ')
        self.df['A'] = self.df['A'].apply(lambda c: ord(c) - 96)

        self.sentence_df = self.fill_sentence()

    def fill_sentence(self):
        filled = []
        for key, row in self.df.iterrows():
            filled_choices = [re.sub('_+', row[c_name], row['Q']) for c_name in self.choice_names]
            filled.append(row[['No', 'A', 'Q']].tolist() + filled_choices)

        df = pd.DataFrame(filled)
        nc = len(self.choice_names)
        df.columns = ['No', 'A', 'Q'] + ['C%d' % i for i in range(1, nc + 1)]
        return df


class SentenceSelection(object):
    def __init__(self, filename, n_chocies=5):
        self.choice_names = ['C%d' % i for i in range(1, n_chocies + 1)]
        path = os.path.join(sc_dir, filename)
        df = pd.read_table(path, sep=',').ix[:, :n_chocies + 3]
        df['ans'] = df['ans'].apply(lambda c: ord(c) - 96)

        df.columns = ['No', 'A', 'Q'] + self.choice_names
        self.sentence_df = df


criterion = nn.CrossEntropyLoss()
# tokenize_ = lambda t: tokenize(t, args.tokenizer)
tokenize_ = lambda t: tokenize(t, 'nltk')


def eval_text(model, _word_ids, bidirec, start_loc=0, end_loc=0):
    # Turn on evaluation mode which disables dropout.
    word_ids_t = torch.LongTensor(_word_ids)
    if args.cuda:
        word_ids_t = word_ids_t.cuda()

    data = Variable((word_ids_t if bidirec else word_ids_t[:-1]).unsqueeze(1), volatile=True)
    target = Variable(word_ids_t[1:-1] if bidirec else word_ids_t[1:])

    output, __ = model(data)
    output_v = output.view(-1, model.ntokens)

    if 0 <= start_loc < end_loc < len(_word_ids):
        partial_out = output_v[start_loc:end_loc, :]
        partial_target = target[start_loc:end_loc]
        return criterion(partial_out, partial_target).data[0]
    else:
        return criterion(output_v, target).data[0]


def eval_model(path, sc, verbose=2):
    checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
    bidirec = checkpoint['args'].bidirec
    eos = checkpoint['args'].eos
    with open(os.path.join(save_dir_root, checkpoint['args'].vocab), 'rb') as f:
        vocab = pickle.load(f)

    model = checkpoint['net']
    if args.cuda:
        model.cuda()
    if verbose >= 2:
        print(checkpoint['args'])
        print(model)
    model.eval()

    for idx, row in sc.sentence_df.iterrows():
        losses = []
        for choice_name in sc.choice_names:
            word_tokens = tokenize_(row[choice_name])
            word_ids = [vocab.word_to_id(token) for token in word_tokens]
            Vocabulary.append_eos(word_ids, eos, eos and bidirec)

            total_loss = eval_text(model, word_ids, bidirec=bidirec)
            losses.append(total_loss)

        pred = np.argmin(losses) + 1
        sc.sentence_df.loc[idx, 'pred'] = pred
        if verbose >= 2:
            print(row['No'], row['A'], pred, np.round(losses, 2), sep=' ')

    accuracy = np.mean([sc.sentence_df.A == sc.sentence_df.pred])
    print("{} {:4.2f}%".format(path, 100 * accuracy))


if __name__ == "__main__":
    # # MSR challenge dataset
    # sc = SentenceCompletion('testing_data.csv', 'test_answer.csv')
    # SAT sentence completion questions
    sc = SentenceSelection('SAT_set_filled.csv')
    sc.sentence_df['pred'] = 0

    save_dir = os.path.join(save_dir_root, args.save_dir)
    if args.ckpt:
        eval_model(os.path.join(save_dir, args.ckpt[0]), sc, verbose=2)
    else:
        ckpt_paths = glob.glob(os.path.join(save_dir, '*.pt'))
        ckpt_paths.sort(key=os.path.getmtime)
        for ckpt_path in ckpt_paths:
            eval_model(ckpt_path, sc, verbose=1)
