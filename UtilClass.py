import torch
import torch.nn as nn
import allennlp.modules.seq2vec_encoders
from allennlp.nn.util import sort_batch_by_length
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

def bottle(v):
    return v.view(-1, v.size(2)) if v is not None else None

def unbottle(v, batch_size):
    return v.view(batch_size, -1, v.size(1))

def shiftLeft(t, pad):
  shifted_t = t[:, 1:] # first dim is batch
  padding =  torch.zeros(t.size(0), 1).fill_(pad).long().cuda()
  return torch.cat((shifted_t, padding), 1)

class ConfigurationError(Exception):
    """
    The exception raised by any AllenNLP object when it's misconfigured
    (e.g. missing properties, invalid properties, unknown properties).
    """
    def __init__(self, message):
        super(ConfigurationError, self).__init__()
        self.message = message

    def __str__(self):
        return repr(self.message)


class ProperLSTM(nn.LSTM):

  def forward(self, seq, seq_lens):
    if not self.batch_first:
      raise ConfigurationError("Our encoder semantics assumes batch is always first!")

    non_zero_length_mask = seq_lens.ne(0).float()
    # make zero lengths into length=1
    seq_lens = seq_lens + seq_lens.eq(0).float()

    sorted_inputs, sorted_sequence_lengths, restoration_indices, sorting_indices =\
                sort_batch_by_length(seq, seq_lens)

    packed_input = pack(sorted_inputs, sorted_sequence_lengths.data.long().tolist(), batch_first=True)
    outputs, final_states = super(ProperLSTM, self).forward(packed_input)

    unpacked_sequence, _ = unpack(outputs, batch_first=True)
    outputs = unpacked_sequence.index_select(0, restoration_indices)
    new_unsorted_states = [self.fix_hidden(state.index_select(1, restoration_indices))
                                                          for state in final_states]

    # To deal with zero length inputs
    outputs = outputs * non_zero_length_mask.view(-1, 1, 1).expand_as(outputs)
    new_unsorted_states[0] = new_unsorted_states[0] * non_zero_length_mask.view(1, -1, 1).expand_as(new_unsorted_states[0])
    new_unsorted_states[1] = new_unsorted_states[1] * non_zero_length_mask.view(1, -1, 1).expand_as(new_unsorted_states[1])

    return outputs, new_unsorted_states

  def fix_hidden(self, h):
    # (layers*directions) x batch x dim to layers x batch x (directions*dim))
    if self.bidirectional:
      h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
    return h

class Bottle(nn.Module):
        def forward(self, input):
            if len(input.size()) <= 2:
                return super(Bottle, self).forward(input)
            size = input.size()[:2]
            out = super(Bottle, self).forward(input.view(size[0]*size[1], -1))
            return out.contiguous().view(size[0], size[1], -1)

class BottleLinear(Bottle, nn.Linear):
    pass

class BottleEmb(nn.Module):
  def forward(self, input):
    size = input.size()
    if len(size) <= 2:
      return super(BottleEmb, self).forward(input)
    if len(size) == 3:
      out = super(BottleEmb, self).forward(input.view(size[0]*size[1], -1))
      return out.contiguous().view(size[0], size[1], size[2], -1)
    elif len(size) == 4:
      out = super(BottleEmb, self).forward(input.view(size[0]*size[1]*size[2], -1))
      return out.contiguous().view(size[0], size[1], size[2], size[3], -1)

class BottleLSTMHelper(nn.Module):
  def forward(self, input, lengths):
    size = input.size()
    if len(size) <= 3:
      return super(BottleLSTMHelper, self).forward(input, lengths)
    if len(size) == 4:
      out = super(BottleLSTMHelper, self).forward(input.view(size[0]*size[1], size[2], -1), lengths.view(-1))
      return (out[0].contiguous().view(size[0], size[1], size[2], -1),
              (out[1][0].contiguous().view(out[1][0].size(0), size[0], size[1], -1),
              out[1][1].contiguous().view(out[1][1].size(0), size[0], size[1], -1))
              )

class BottleLSTM(BottleLSTMHelper, ProperLSTM):
    pass

class BottleEmbedding(BottleEmb, nn.Embedding):
    pass
