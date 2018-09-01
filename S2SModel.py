import torch
from torch.autograd import Variable
import torch.nn as nn
from Statistics import Statistics
from UtilClass import bottle
from Beam import Beam
from CopyGenerator import CopyGenerator, RegularGenerator, ProdGenerator
from RegularDecoder import RegularDecoder
from ProdDecoder import ProdDecoder
from RegularEncoder import RegularEncoder 
from ConcodeEncoder import ConcodeEncoder
from ConcodeDecoder import ConcodeDecoder
from decoders import DecoderState

class S2SModel(nn.Module):
  def __init__(self, opt, vocabs):
    super(S2SModel, self).__init__()

    self.opt = opt
    self.vocabs = vocabs

    if self.opt.encoder_type == "regular":
      self.encoderClass = RegularEncoder
    elif self.opt.encoder_type == "concode":
      self.encoderClass = ConcodeEncoder
    self.encoder = self.encoderClass(vocabs, opt)

    if self.opt.decoder_type == "prod":
      self.decoderClass = ProdDecoder
    elif self.opt.decoder_type == "concode":
      self.decoderClass = ConcodeDecoder
    else:
      self.decoderClass = RegularDecoder

    self.decoder = self.decoderClass(vocabs, opt)

    if self.opt.decoder_type in ["prod", "concode"]:
      generator = ProdGenerator
    elif self.opt.copy_attn:
      generator = CopyGenerator
    else:
      generator = RegularGenerator

    self.generator = generator(self.opt.decoder_rnn_size, vocabs, self.opt)
    self.cuda()


  def forward(self, batch):
    # initial parent states for Prod Decoder
    batch_size = batch['seq2seq'].size(0)
    batch['parent_states'] = {}
    for j in range(0, batch_size):
      batch['parent_states'][j] = {}
      if self.opt.decoder_type in ["prod", "concode"]:
        batch['parent_states'][j][0] = Variable(torch.zeros(1, 1, self.opt.decoder_rnn_size).cuda(), requires_grad=False)

    context, context_lengths, enc_hidden = self.encoder(batch)

    decInitState = DecoderState(enc_hidden, Variable(torch.zeros(batch_size, 1, self.opt.decoder_rnn_size).cuda(), requires_grad=False))

    output, attn, copy_attn = self.decoder(batch, context, context_lengths, decInitState)

    del batch['parent_states']


    # Other generators will not use the extra parameters
    # Let the generator put the src_map in cuda if it uses it
    # TODO: Make sec_map variable again in generator
    src_map = torch.zeros(0, 0)
    if self.opt.var_names:
      src_map = torch.cat((src_map, batch['concode_src_map_vars']), 1)
    if self.opt.method_names:
      src_map = torch.cat((src_map, batch['concode_src_map_methods']), 1)

    scores = self.generator(bottle(output), bottle(copy_attn), src_map if self.opt.encoder_type in ["concode"] else  batch['src_map'], batch)
    loss, total, correct = self.generator.computeLoss(scores, batch)

    return loss, Statistics(loss.data[0], total, correct, self.encoder.n_src_words)

  # This only works for a batch size of 1
  def predict(self, batch, opt, vis_params):
      curr_batch_size = batch['seq2seq'].size(0)
      assert(curr_batch_size == 1)
      context, context_lengths, enc_hidden = self.encoder(batch)
      return self.decoder.predict(enc_hidden, context, context_lengths, batch, opt.beam_size, opt.max_sent_length, self.generator, opt.replace_unk, vis_params) 
