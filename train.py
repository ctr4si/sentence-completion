import os
import time
import math
import pickle
import logging

import numpy as np
import torch
from torch import nn, optim
from torch.autograd import Variable

from config import save_dir_root, args
from input import get_txts, Vocabulary, LineInput, TextInput
from model import WordRNN, WordbiRNN, SampledSoftmax


logging.basicConfig(format='%(asctime)s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S', level=logging.DEBUG)

logger = logging.getLogger(__name__)
logger.propagate = False
console_handler = logging.StreamHandler()
logger.addHandler(console_handler)

logging.debug("running {}".format(__file__))

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if args.cuda and torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)

# Use pre built vocabulary if it exists
vocab_path = '{}{}'.format(save_dir_root, args.vocab)
if os.path.exists(vocab_path):
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    update = False
else:
    vocab = Vocabulary(args.lower, args.min_cnt)
    update =True

tr_txts = get_txts(args.corpus, 'train')
va_txts = get_txts(args.corpus, 'valid')
if args.bptt > 0:
    tr_input = TextInput(tr_txts, vocab, args.tokenizer, update)
    va_input = TextInput(va_txts, vocab, args.tokenizer, update)
    batchify = (lambda mode, bsz, seed: (tr_input if mode == 'tr' else va_input)
                .batchify(bsz, args.bptt, True, seed))
else:
    eos = args.eos
    bos = args.eos and args.bidirec
    tr_input = LineInput(tr_txts, vocab, args.tokenizer, update, bos, eos,
                         args.min_len, args.max_len)
    va_input = LineInput(va_txts, vocab, args.tokenizer, update, bos, eos,
                         args.min_len, args.max_len)
    batchify = (lambda mode, bsz, seed: (tr_input if mode == 'tr' else va_input)
                .batchify(bsz, seed))

ntokens = len(vocab)
logging.debug("{} word tokens".format(ntokens))

# Save vocabulary
args.vocab = 'vocab{}.pkl'.format(ntokens)
vocab_path = os.path.join(save_dir_root, args.vocab)
if not os.path.exists(vocab_path):
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)

val_data = batchify('va', args.eval_batch_size, False)

# Build the model
if args.resume:
    ckpt_path = os.path.join(save_dir_root, args.resume)
    checkpoint = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
    model = checkpoint['net']
    learnables = list(filter(lambda p: p.requires_grad, model.parameters()))
    optimizer = checkpoint['optimizer']
    logging.debug("loaded from {} successfully".format(ckpt_path))
    logging.debug(optimizer.state_dict())
else:
    if args.bidirec:
        model = WordbiRNN(args.rnncell, ntokens, args.nlayers,
                          args.nhid, args.emsize, args.outsize,
                          args.dropout, args.pad_idx, args.tied)
    else:
        model = WordRNN(args.rnncell, ntokens, args.nlayers,
                        args.nhid, args.emsize, args.outsize,
                        args.dropout, args.pad_idx)
    learnables = list(filter(lambda p: p.requires_grad, model.parameters()))
    optimizer = getattr(optim, args.optim)(learnables, lr=args.lr)

if args.nsampled > ntokens:
    logging.debug("adjust # negative samples from {} to {}"
                  .format(args.nsampled, ntokens))
    args.nsampled = ntokens
ss = SampledSoftmax(ntokens, args.nsampled, model.decoder)
criterion = nn.CrossEntropyLoss(size_average=False)
if args.cuda:
    model.cuda()
logging.debug(model)


def get_xy(source, evaluation=False):
    source_t = torch.LongTensor(source).t().contiguous()
    if args.cuda:
        source_t = source_t.cuda()
    data = Variable(source_t if args.bidirec else source_t[:-1], volatile=evaluation)
    target = Variable((source_t[1:-1] if args.bidirec else source_t[1:]).view(-1))
    return data, target


def evaluate(data_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss, total_len = 0, 0
    for batch_data in data_source:
        data, targets = get_xy(batch_data, evaluation=True)
        output, __ = model(data)
        output_flat = output.view(-1, ntokens)
        total_loss += criterion(output_flat, targets).data
        total_len += len(targets)
    return total_loss[0] / total_len


def train(shuffle):
    # Turn on training mode which enables dropout.
    model.train()
    seed = int(torch.rand(1).numpy() * np.iinfo(np.uint32).max) if shuffle else False
    bsz = args.batch_size
    train_data = batchify('tr', bsz, seed)

    total_loss, total_len = 0, 0
    start_time = time.time()
    nb = len(train_data)
    for batch, batch_data in enumerate(train_data, start=1):
        data, targets = get_xy(batch_data)

        model.zero_grad()
        if args.nsampled > 0:
            output, __ = model(data, decode=False)
            logits, new_targets = ss(output.view(targets.size(0), -1), targets)
            loss = criterion(logits, new_targets)
        else:
            logits, __ = model(data)
            loss = criterion(logits.view(-1, ntokens), targets)

        total_loss += loss.data[0]
        total_len += len(targets)

        loss /= bsz
        loss.backward()
        if args.clip > 0:
            torch.nn.utils.clip_grad_norm(learnables, args.clip)
        optimizer.step()

        remainder = batch % args.log_interval
        if remainder == 0 or batch == nb:
            cur_loss = total_loss / total_len
            denom = (args.log_interval if remainder == 0 else remainder)
            time_per_batch = (time.time() - start_time) * 1000 / denom
            logger.info('| epoch {:2d} | {:5d}/{:5d} batches '
                        '| ms/batch {:5.2f} | loss {:5.2f} | ppl {:8.2f} |'
                        .format(epoch, batch, nb, time_per_batch,
                                cur_loss, math.exp(cur_loss)))
            total_loss, total_len = 0, 0
            start_time = time.time()

# Loop over epochs.
best_val_loss = None

save_dir = os.path.join(save_dir_root, args.save_dir)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
save_path = os.path.join(save_dir, 'model.pt')

file_handler = logging.FileHandler(os.path.join(save_dir, 'log.txt'), mode='w')
file_handler.setLevel(logging.INFO)
logger.addHandler(file_handler)

# At any point you can hit Ctrl + C to break out of training early.
try:
    logger.info('-' * 89)
    for epoch in range(args.start_epoch, args.epochs+1):
        epoch_start_time = time.time()
        train(args.shuffle)
        val_loss = evaluate(val_data)
        logger.info('-' * 89)
        logger.info('| end of epoch {:3d} | time: {:5.2f}s '
                    '| valid loss {:5.2f} | valid ppl {:8.2f} |'
                    .format(epoch, (time.time() - epoch_start_time),
                            val_loss, math.exp(val_loss)))
        logger.info('-' * 89)

        breaking_best = not best_val_loss or val_loss < best_val_loss
        if epoch >= args.decay_after > 0:
            if (epoch - args.decay_after) % args.decay_period == 0:
                for group in optimizer.param_groups:
                    group['lr'] *= args.decay_rate

        # Save the model if the validation loss is the best we've seen so far.
        if (epoch % args.save_period == 0) and (breaking_best or args.save_all):
            checkpoint = {'net': model, 'optimizer': optimizer,
                          'epoch': epoch, 'args': args}
            if args.save_all:
                save_path = os.path.join(save_dir, 'ep{}.pt'.format(epoch))
            torch.save(checkpoint, save_path)

            if breaking_best:
                best_val_loss = val_loss

except KeyboardInterrupt:
    logger.debug('-' * 89)
    logger.debug('Exiting from training early')

