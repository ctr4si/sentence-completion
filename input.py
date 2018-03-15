import os
import string
import logging
from collections import Counter

from tqdm import tqdm
import numpy as np
from config import prepro_dir, gbw_ho_dir, gbw_tr_dir


def tokenize(text, tokenizer):
    if tokenizer == 'nltk':
        import nltk
        return nltk.tokenize.word_tokenize(text)
   else:
        return text.split()


def get_txts(name, mode):
    if name in ('guten', 'sejong', 'sejong_pos'):
        return [os.path.join(prepro_dir, name, '{}.txt'.format(mode))]
    elif name == 'gbw' and mode == 'train':
        return [os.path.join(gbw_tr_dir, f) for f in os.listdir(gbw_tr_dir)]
    elif name == 'gbw' and mode == 'valid':
        return [os.path.join(gbw_ho_dir, 'news.en-00000-of-00100')]
        # return [os.path.join(gbw_ho_dir, 'news.en.heldout-00000-of-00050')]


class Vocabulary(object):

    PAD = 0
    UNK = 1
    BLANK = 2
    BOS = 3
    EOS = 4
    PAD_TK = '<pad>'
    UNK_TK = '<unk>'
    BLANK_TK = '<blank>'
    BOS_TK = '<bos>'
    EOS_TK = '<eos>'

    def __init__(self, lower=False, min_cnt=1):
        # self.counter = nltk.FreqDist()
        self.counter = Counter()
        self.lower = lower
        self.min_cnt = min_cnt
        self.init()

    def __len__(self):
        return len(self.words)

    def init(self):
        self.word2id = {}
        self.words = []
        self.add_word(Vocabulary.PAD_TK)
        self.add_word(Vocabulary.UNK_TK)
        self.add_word(Vocabulary.BLANK_TK)
        self.add_word(Vocabulary.BOS_TK)
        self.add_word(Vocabulary.EOS_TK)

    def update_counter(self, words):
        self.counter.update(words)

    def add_word(self, word):
        if self.lower:
            word = word.lower()
        if word not in self.word2id:
            self.words.append(word)
            self.word2id[word] = len(self.words) - 1
        return self.word2id[word]

    def build_from_counter(self, min_cnt=None):
        if min_cnt is None:
            min_cnt = self.min_cnt
        for w, cnt in self.counter.most_common():
            if cnt >= max(min_cnt, 1):
                self.add_word(w)

    def word_to_id(self, word):
        if self.lower:
            word = word.lower()
        return self.word2id.get(word, Vocabulary.UNK)

    @staticmethod
    def append_eos(word_ids, eos=True, bos=False):
        if bos:
            word_ids.insert(0, Vocabulary.BOS)
        if eos:
            word_ids.append(Vocabulary.EOS)


class TextInput(object):
    def __init__(self, txt_paths, vocab, tokenizer='split', update=False):
        self.tokenizer = tokenizer
        if update:
            for word_tokens in self.tokenize_txt(txt_paths):
                vocab.update_counter(word_tokens)
            vocab.build_from_counter()

        self.txt_paths = txt_paths
        self.vocab = vocab
        self.data = None

    def load_data(self):
        word_ids = []
        for word_tokens in self.tokenize_txt(self.txt_paths):
            word_ids.append([self.vocab.word_to_id(token) for token in word_tokens])
        return word_ids

    def tokenize_txt(self, txt_paths):
        for txt_path in tqdm(txt_paths, ncols=70):
            with open(txt_path, 'r') as f:
                contents = f.read()
                yield tokenize(contents, self.tokenizer)

    def batchify(self, bsz, bptt, use_remainder=True, shuffle=False):
        if self.data is None:
            self.data = self.load_data()
        np.random.seed(shuffle)

        sequences = []
        remainder = 0
        for word_ids in self.data:
            offset = 0 if use_remainder else np.random.randint(bptt)
            sequences.extend(word_ids[i:i + bptt] for i in
                             range(offset, len(word_ids) - bptt + 1, bptt))

            curr_remainder = (len(word_ids) - offset) % bptt
            if use_remainder and len(word_ids) >= bptt and curr_remainder > 0:
                sequences.append(word_ids[-bptt:])
            else:
                remainder += (offset + curr_remainder)

        if shuffle:
            np.random.shuffle(sequences)
        batches = [sequences[i:i + bsz] for i in
                   range(0, len(sequences) - bsz + 1, bsz)]
        remainder += bptt * (len(sequences) % bsz)

        logging.debug("discarded {} tokens resulting in {} batches "
                      .format(remainder, len(batches)))
        return batches


class LineInput(object):
    def __init__(self, txt_paths, vocab, tokenizer='split', update=False,
                 bos=False, eos=False, min_len=1, max_len=120):
        self.tokenizer = tokenizer
        if update:
            for word_tokens in self.tokenize_txt(txt_paths, min_len, max_len):
                vocab.update_counter(word_tokens)
            vocab.build_from_counter()

        self.txt_paths = txt_paths
        self.vocab = vocab
        self.bos = bos
        self.eos = eos
        self.min_len = min_len
        self.max_len = max_len
        self.data = None

    def load_data(self):
        len2lines = dict()
        for word_tokens in self.tokenize_txt(self.txt_paths, self.min_len, self.max_len):
            word_ids = [self.vocab.word_to_id(token) for token in word_tokens]
            Vocabulary.append_eos(word_ids, self.eos, self.bos)

            lines = len2lines.get(len(word_ids), [])
            lines.append(word_ids)
            len2lines[len(word_ids)] = lines
        return len2lines

    def tokenize_txt(self, txt_paths, min_len, max_len):
        for txt_path in tqdm(txt_paths, ncols=70):
            with open(txt_path, 'r') as f:
                # assume that each line is a sentence
                for line in f.readlines():
                    word_tokens = tokenize(line, self.tokenizer)
                    if min_len <= len(word_tokens) <= max_len:
                        yield word_tokens

    def batchify(self, bsz, shuffle=False):
        if self.data is None:
            self.data = self.load_data()
        np.random.seed(shuffle)

        batches = []
        n_lines, remainder = 0, 0
        for len_, lines in self.data.items():
            n_lines += len(lines)
            if shuffle:
                np.random.shuffle(lines)
            nb = len(lines) // bsz
            remainder += len(lines) % bsz
            if nb > 0:
                batches.extend([lines[i*bsz:(i+1)*bsz] for i in range(nb)])

        np.random.shuffle(batches)
        logging.debug("discarded {} of {} lines resulting in {} batches "
                      .format(remainder, n_lines, len(batches)))
        return batches
