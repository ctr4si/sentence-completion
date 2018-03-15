import os
import argparse


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

parser = argparse.ArgumentParser()

# source path
save_dir_root = 'train/'
prepro_dir = 'data/prepro/'
gbw_dir = 'data/1-billion-word-language-modeling-benchmark-r13output/'
gbw_tr_dir = os.path.join(gbw_dir, 'training-monolingual.tokenized.shuffled')
gbw_ho_dir = os.path.join(gbw_dir, 'heldout-monolingual.tokenized.shuffled')
sc_dir = 'data/completion/'

# model structure
parser.add_argument('--rnncell', type=str, default='LSTM')
parser.add_argument('--emsize', type=int, default=200)
parser.add_argument('--nhid', type=int, default=600)
parser.add_argument('--outsize', type=int, default=400)
parser.add_argument('--nlayers', type=int, default=2)
parser.add_argument('--bidirec', type=str2bool, default=True)
parser.add_argument('--tied', action='store_true')

# data
parser.add_argument('--corpus', type=str, default='guten')
parser.add_argument('--min_len', type=int, default=10)
parser.add_argument('--max_len', type=int, default=80)

# vocabulary
parser.add_argument('--vocab', type=str, default=None)
parser.add_argument('--tokenizer', type=str, default='split')
parser.add_argument('--lower', type=str2bool, default=True)
parser.add_argument('--min_cnt', type=int, default=6)
parser.add_argument('--eos', action='store_true')

# training
parser.add_argument('--pad_idx', type=int, default=0)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--nsampled', type=int, default=-1)

parser.add_argument('--shuffle', type=str2bool, default=True)
parser.add_argument('--seed', type=int, default=3333)
parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--eval_batch_size', type=int, default=10)
parser.add_argument('--bptt', type=int, default=-1)

parser.add_argument('--optim', type=str, default='SGD')
parser.add_argument('--lr', type=float, default=.5)
parser.add_argument('--clip', type=float, default=5.0)
parser.add_argument('--decay_after', type=int, default=5)
parser.add_argument('--decay_rate', type=float, default=0.5)
parser.add_argument('--decay_period', type=int, default=1)
parser.add_argument('--epochs', type=int, default=10)

parser.add_argument('--save_dir', type=str, default='test')
parser.add_argument('--log_interval', type=int, default=10000)
parser.add_argument('--eval_interval', type=int, default=10000)
parser.add_argument('--save_all', type=str2bool, default=True)

parser.add_argument('--resume', type=str, default=None)
parser.add_argument('--start_epoch', type=int, default=1)
parser.add_argument('--save_period', type=int, default=1)

# gpu
parser.add_argument('--cuda', action='store_true')

# eval
parser.add_argument('--ckpt', type=str, nargs='+', default=None)

args = parser.parse_args()
