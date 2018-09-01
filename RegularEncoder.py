from torch import nn
from torch.autograd import Variable
from UtilClass import ProperLSTM

class RegularEncoder(nn.Module):
  def __init__(self, vocabs, opt):
    super(RegularEncoder, self).__init__()
    self.opt = opt
    self.vocabs = vocabs

    self.encoder_embedding = nn.Embedding(
      len(self.vocabs['seq2seq']),
      self.opt.src_word_vec_size,
      padding_idx=self.vocabs['seq2seq'].stoi['<blank>'])

    self.encoder_rnn = ProperLSTM(
      input_size=self.opt.src_word_vec_size,
      hidden_size=(self.opt.rnn_size // 2 if self.opt.brnn else self.opt.rnn_size),
      num_layers=self.opt.enc_layers,
      dropout=self.opt.dropout,
      bidirectional=self.opt.brnn,
      batch_first=True)

  def forward(self, batch):
    src = Variable(batch['seq2seq'].cuda(), requires_grad=False)
    src_embeddings = self.encoder_embedding(src)

    lengths = src.ne(self.vocabs['seq2seq'].stoi['<blank>']).float().sum(1)
    self.n_src_words = lengths.sum().data[0]
    context, enc_hidden = self.encoder_rnn(src_embeddings, lengths)
    return context, lengths, enc_hidden
