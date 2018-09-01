import torch
from torch import nn
from GlobalAttention import GlobalAttention
from torch.autograd import Variable
from Beam import Beam
from decoders import DecoderState, Prediction

class RegularDecoder(nn.Module):
  def __init__(self, vocabs, opt):
    super(RegularDecoder, self).__init__()

    self.opt = opt
    self.vocabs = vocabs

    self.decoder_embedding = nn.Embedding(
      len(vocabs['code']),
      opt.tgt_word_vec_size,
      padding_idx=vocabs['code'].stoi['<blank>'])

    self.attn = GlobalAttention(
        opt.rnn_size,
        attn_type='general')

    if opt.copy_attn:
      self.copy_attn = GlobalAttention(
          opt.rnn_size,
          attn_type='general')

    self.decoder_rnn = nn.LSTM(
      input_size=opt.tgt_word_vec_size,
      hidden_size=opt.rnn_size,
      num_layers=opt.dec_layers,
      dropout=opt.dropout,
      batch_first=True)

    self.decoder_dropout = nn.Dropout(opt.dropout)

  def forward(self, batch, context, context_lengths, decState):
    inp = Variable(batch['code'].cuda(), requires_grad=False)
    tgt_embeddings = self.decoder_embedding(inp)

    rnn_output, prev_hidden = self.decoder_rnn(tgt_embeddings, decState.hidden)
    rnn_output.contiguous()
    attn_output, attn_scores = self.attn(rnn_output, context, context_lengths)
    attn_output = self.decoder_dropout(attn_output)
    decState.update_state(prev_hidden, attn_output)
    copy_attn_scores = None
    if self.opt.copy_attn:
      _, copy_attn_scores = self.copy_attn(attn_output, context, context_lengths)
    output = attn_output

    return output, attn_scores, copy_attn_scores

  def predict(self, enc_hidden, context, context_lengths, batch, beam_size, max_code_length, generator, replace_unk, vis_params):

      decState = DecoderState(
        enc_hidden,
        Variable(torch.zeros(1, 1, self.opt.rnn_size).cuda(), requires_grad=False)
      )

      # Repeat everything beam_size times.
      def rvar(a, beam_size):
        return Variable(a.repeat(beam_size, 1, 1), volatile=True)
      context = rvar(context.data, beam_size)
      context_lengths = context_lengths.repeat(beam_size)
      decState.repeat_beam_size_times(beam_size)

      beam = Beam(beam_size,
                      cuda=True,
                      vocab=self.vocabs['code'])

      for i in range(max_code_length):
        if beam.done():
          break

        # Construct batch x beam_size nxt words.
        # Get all the pending current beam words and arrange for forward.
        # Uses the start symbol in the beginning
        inp = beam.getCurrentState() # Should return a batch of the frontier
        # Turn any copied words to UNKs
        if self.opt.copy_attn:
            inp['code'] = inp['code'].masked_fill_(inp['code'].gt(len(self.vocabs["code"]) - 1), self.vocabs["code"].stoi['<unk>'])
        # Run one step., decState gets automatically updated
        decOut, attn, copy_attn = self.forward(inp, context, context_lengths, decState)

        # decOut: beam x rnn_size
        decOut = decOut.squeeze(1)

        out = generator(decOut, copy_attn.squeeze(1) if copy_attn is not None else None, batch['src_map'], inp).data
        out = out.unsqueeze(1)
        if self.opt.copy_attn:
          out = generator.collapseCopyScores(out, batch)
          out = out.log()

        # beam x tgt_vocab
        beam.advance(out[:, 0], attn.data[:, 0])
        decState.beam_update(beam.getCurrentOrigin(), beam_size)

      score, times, k = beam.getFinal() # times is the length of the prediction
      hyp, att = beam.getHyp(times, k)
      goldNl = self.vocabs['seq2seq'].addStartOrEnd(batch['raw_seq2seq'][0])
      goldCode = self.vocabs['code'].addStartOrEnd(batch['raw_code'][0])
      predSent = self.buildTargetTokens(
        hyp,
        self.vocabs,
        goldNl,
        att,
        batch['seq2seq_vocab'][0],
        replace_unk
      )
      return Prediction(goldNl, goldCode, predSent, att)

  def buildTargetTokens(self, pred, vocabs, src, attn, copy_vocab, replace_unk):
    vocab = vocabs['code']
    tokens = []
    for tok in pred:
        if tok < len(vocab):
            tokens.append(vocab.itos[tok])
        else:
            tokens.append(copy_vocab.itos[tok - len(vocab)])
        if tokens[-1] == '</s>':
            tokens = tokens[:-1]
            break

    if replace_unk and attn is not None:
        for i in range(len(tokens)):
            if tokens[i] == '<unk>':
                _, maxIndex = attn[i].max(0)
                tokens[i] = src[maxIndex[0]]

    return tokens
