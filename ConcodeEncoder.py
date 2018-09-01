import torch
from torch import nn
from torch.autograd import Variable
from UtilClass import ProperLSTM, BottleEmbedding, BottleLSTM

class ConcodeEncoder(nn.Module):

  def __init__(self, vocabs, opt):
    super(ConcodeEncoder, self).__init__()

    self.opt = opt
    self.vocabs = vocabs

    self.names_embedding = BottleEmbedding(
      len(vocabs['names_combined']),
      self.opt.src_word_vec_size * 2,
      padding_idx=self.vocabs['names_combined'].stoi['<blank>'])

    self.types_embedding = BottleEmbedding(
      len(vocabs['types']),
      self.opt.src_word_vec_size * 2)

    self.src_rnn = ProperLSTM(
      input_size=self.opt.src_word_vec_size * 2,
      hidden_size=(self.opt.decoder_rnn_size // 2 if self.opt.brnn else self.opt.decoder_rnn_size),
      num_layers=self.opt.enc_layers,
      dropout=self.opt.dropout,
      bidirectional=self.opt.brnn,
      batch_first=True)

    self.camel_rnn = BottleLSTM(
      input_size=self.opt.src_word_vec_size * 2,
      hidden_size=(self.opt.decoder_rnn_size // 2 if self.opt.brnn else self.opt.src_word_vec_size),
      num_layers=self.opt.enc_layers,
      dropout=self.opt.dropout,
      bidirectional=self.opt.brnn,
      batch_first=True)

    self.var_rnn = BottleLSTM(
      input_size=self.opt.src_word_vec_size * 2,
      hidden_size=(self.opt.decoder_rnn_size // 2 if self.opt.brnn else self.opt.rnn_size),
      num_layers=self.opt.enc_layers,
      dropout=self.opt.dropout,
      bidirectional=self.opt.brnn,
      batch_first=True)

    self.method_rnn = BottleLSTM(
      input_size=self.opt.src_word_vec_size * 2,
      hidden_size=(self.opt.decoder_rnn_size // 2 if self.opt.brnn else self.opt.rnn_size),
      num_layers=self.opt.enc_layers,
      dropout=self.opt.dropout,
      bidirectional=self.opt.brnn,
      batch_first=True)

  def forward(self, batch):
    batch_size = batch['src'].size(0)

    src = Variable(batch['src'].cuda(), requires_grad=False)
    src_embeddings = self.names_embedding(src)
    lengths = src.ne(self.vocabs['names_combined'].stoi['<blank>']).float().sum(1)
    self.n_src_words = lengths.sum().data[0]
    context, enc_hidden = self.src_rnn(src_embeddings, lengths)
    src_context = context

    if self.opt.var_names:
      # varcamel is b x vlen x camel_len
      varCamel = Variable(batch['varNames'].transpose(1, 2).contiguous().cuda(), requires_grad=False)
      if self.opt.nocamel:
        varCamel = varCamel[:, :, 0:1].contiguous()


      varCamel_lengths = varCamel.ne(self.vocabs['names_combined'].stoi['<blank>']).float().sum(2)
      varNames_camel_embeddings = self.names_embedding(varCamel)
      varCamel_context, varCamel_encoded = self.camel_rnn(varNames_camel_embeddings, varCamel_lengths)

      varTypes = Variable(batch['varTypes'].cuda(), requires_grad=False)
      varTypes_embeddings = self.types_embedding(varTypes)
      var_input = torch.cat((varTypes_embeddings.unsqueeze(2), varCamel_encoded[0][1, :].unsqueeze(2)), 2)
      var_lengths = varTypes.ne(self.vocabs['types'].stoi['<blank>']).float() * 2

      var_context, var_hidden = self.var_rnn(var_input, var_lengths)

      var_context = var_context.view(var_context.size(0), -1, var_context.size(3)) # interleave type and name, type first

    if self.opt.method_names:
      methodCamel = Variable(batch['methodNames'].transpose(1, 2).contiguous().cuda(), requires_grad=False)
      if self.opt.nocamel:
        methodCamel = methodCamel[:, :, 0:1].contiguous()
      methodCamel_lengths = methodCamel.ne(self.vocabs['names_combined'].stoi['<blank>']).float().sum(2)
      methodNames_camel_embeddings = self.names_embedding(methodCamel)
      methodCamel_context, methodCamel_encoded = self.camel_rnn(methodNames_camel_embeddings, methodCamel_lengths)

      methodReturns = Variable(batch['methodReturns'].cuda(), requires_grad=False)
      methodReturns_embeddings = self.types_embedding(methodReturns)
      method_input = torch.cat((methodReturns_embeddings.unsqueeze(2), methodCamel_encoded[0][1,:].unsqueeze(2)), 2)
      method_lengths = methodReturns.ne(self.vocabs['types'].stoi['<blank>']).float() * 2

      method_context, method_hidden = self.method_rnn(method_input, method_lengths)
      method_context = method_context.view(method_context.size(0), -1, method_context.size(3)) # interleave type and name, type first

    (batch_size, max_var_len) = batch['varTypes'].size()
    (batch_size, max_method_len) = batch['methodReturns'].size()

    src_attention_mask = Variable(batch['src'].ne(self.vocabs['names_combined'].stoi['<blank>']).cuda(), requires_grad=False)
    var_attention_mask = Variable(batch['varTypes'].ne(self.vocabs['types'].stoi['<blank>']).unsqueeze(2).expand(batch_size, max_var_len, 2).contiguous().view(batch_size, -1).cuda(), requires_grad=False)
    method_attention_mask = Variable(batch['methodReturns'].ne(self.vocabs['types'].stoi['<blank>']).unsqueeze(2).expand(batch_size, max_method_len, 2).contiguous().view(batch_size, -1).cuda(), requires_grad=False)

    ret_context = [src_context]
    ret_mask = [src_attention_mask]
    if self.opt.var_names:
      ret_context.append(var_context)
      ret_mask.append(var_attention_mask)
    if self.opt.method_names:
      ret_context.append(method_context)
      ret_mask.append(method_attention_mask)
    return tuple(ret_context), tuple(ret_mask), enc_hidden
