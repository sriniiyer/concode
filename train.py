import random
import argparse
import torch
from torch import cuda
import torch.nn as nn
from preprocess import Vocab, CDDataset
from S2SModel import S2SModel
from Trainer import Trainer
import os
import numpy

def main():

  parser = argparse.ArgumentParser(description='train.py')

  parser.add_argument("-num_gpus", type=int)
  parser.add_argument("-max_camel", type=int, default=20)
  parser.add_argument('-src_word_vec_size', type=int, default=500,
                      help='Src word embedding sizes')
  parser.add_argument('-tgt_word_vec_size', type=int, default=500,
                      help='Tgt word embedding sizes')
  parser.add_argument('-enc_layers', type=int, default=1,
                      help='Number of layers in the encoder')
  parser.add_argument('-dec_layers', type=int, default=1,
                      help='Number of layers in the decoder')
  parser.add_argument('-rnn_size', type=int, default=500,
                      help='Size of LSTM hidden states')
  parser.add_argument('-decoder_rnn_size', type=int, default=1024,
                      help='Size of LSTM hidden states')
  parser.add_argument('-brnn', action="store_true",
                      help="Use a bidirectional RNN in the encoder")
  parser.add_argument('-nocamel', action="store_true",
                      help="Do not split based on camel case.")
  parser.add_argument('-var_names', action="store_true",
                      help="Dont use variables.")
  parser.add_argument('-twostep', action="store_true",
                      help="Dont use 2-step attention.")
  parser.add_argument('-method_names', action="store_true",
                      help="Dont use methods.")
  parser.add_argument('-trunc', type=int, default=-1,
                      help='Truncate training set.')
  parser.add_argument('-copy_attn', action="store_true",
                      help='Train copy attention layer.')
  parser.add_argument('-data', required=True,
                      help="""Path prefix to the ".train.pt" and
                      ".valid.pt" file path from preprocess.py""")
  parser.add_argument('-save_model', default='model',
                      help="""Model filename (the model will be saved as
                      <save_model>_epochN_PPL.pt where PPL is the
                      validation perplexity""")
  # GPU
  parser.add_argument('-seed', type=int, default=-1,
                      help="""Random seed used for the experiments
                      reproducibility.""")

  # Optimization options
  parser.add_argument('-batch_size', type=int, default=1,
                      help='Maximum batch size')
  parser.add_argument('-epochs', type=int, default=30,
                      help='Number of training epochs')
  parser.add_argument('-encoder_type', default='regular',
                      choices=['regular', 'concode'],
                      help="""Encoder Type.""")
  parser.add_argument('-decoder_type', default='regular',
                      choices=['regular', 'prod', 'concode'],
                      help="""Decoder Type.""")
  parser.add_argument('-max_grad_norm', type=float, default=5,
                      help="""If the norm of the gradient vector exceeds this,
                      renormalize it to have the norm equal to
                      max_grad_norm""")
  parser.add_argument('-dropout', type=float, default=0.3,
                      help="Dropout probability; applied in LSTM stacks.")
  # learning rate
  parser.add_argument('-learning_rate', type=float, default=1.0,
                      help="""Starting learning rate. If adagrad/adadelta/adam
                      is used, then this is the global learning rate.
                      Recommended settings: sgd = 1, adagrad = 0.1,
                      adadelta = 1, adam = 0.001""")
  parser.add_argument('-learning_rate_decay', type=float, default=0.8,
                      help="""If update_learning_rate, decay learning rate by
                      this much if (i) perplexity does not decrease on the
                      validation set or (ii) epoch has gone past
                      start_decay_at""")

  parser.add_argument('-report_every', type=int, default=500,
                      help="Print stats at this interval.")
  parser.add_argument('-train_from', default='', type=str,
                      help="""If training from a checkpoint then this is the
                      path to the pretrained model's state_dict.""")
  parser.add_argument('-start_epoch', type=int, default=None,
                      help='The epoch from which to start. Use this together with train_from.')

  opt = parser.parse_args()

  torch.backends.cudnn.deterministic = True
  torch.cuda.set_device(0)
  torch.manual_seed(opt.seed)
  random.seed(opt.seed)
  torch.cuda.manual_seed(opt.seed)
  numpy.random.seed(opt.seed)


  try:
    os.makedirs(opt.save_model)
  except:
    pass

  print('Loading train set\n')
  train = torch.load(opt.data + '.train.pt')
  print('Loaded Train :')
  valid = torch.load(opt.data + '.valid.pt')

  print('Loaded datasets:') 

  if opt.train_from:
    assert(opt.start_epoch)
    print("Train from activated. Ignoring parameters and loading them from model")
    checkpoint = torch.load(opt.train_from, map_location=lambda storage, loc: storage)
    checkpoint['opt'].start_epoch = opt.start_epoch
    checkpoint['opt'].epochs = opt.epochs
    opt = checkpoint['opt']
    opt.prev_optim = checkpoint['optim']
    vocabs = checkpoint['vocab']
    vocabs['mask'] = vocabs['mask'].cuda()
  else:
    vocabs = torch.load(opt.data + '.vocab.pt')

  print(opt)
  model = S2SModel(opt, vocabs)

  if opt.start_epoch:
    model.load_state_dict(checkpoint['model'])

  trainer  = Trainer(model)
  trainer.run_train_batched(train, valid, vocabs)

if __name__ == "__main__":
  main()
