import torch
from torch import nn
from GlobalAttention import GlobalAttention
from torch.autograd import Variable
from Beam import TreeBeam, Beam
from UtilClass import bottle, unbottle
from preprocess import lhs, rhs, CDDataset
from decoders import DecoderState, Prediction

class ProdDecoder(nn.Module):
  def __init__(self, vocabs, opt):
    super(ProdDecoder, self).__init__()

    self.opt = opt
    self.vocabs = vocabs

    self.nt_embedding = nn.Embedding(
      len(vocabs['nt']),
      opt.tgt_word_vec_size,
      padding_idx=vocabs['nt'].stoi['<blank>'])

    self.rule_embedding = nn.Embedding(
      len(vocabs['prev_rules']),
      opt.tgt_word_vec_size,
      padding_idx=vocabs['prev_rules'].stoi['<blank>'])

    self.attn = GlobalAttention(
        opt.rnn_size,
        attn_type='general')

    if opt.copy_attn:
      self.copy_attn = GlobalAttention(
          opt.rnn_size,
          attn_type='general')

    self.decoder_rnn = nn.LSTM(
      input_size=opt.tgt_word_vec_size * 3 + opt.rnn_size, # nt and prev_rule
      hidden_size=opt.rnn_size,
      num_layers=opt.dec_layers,
      dropout=opt.dropout,
      batch_first=True)

    self.decoder_dropout = nn.Dropout(opt.dropout)

  def forward(self, batch, context, context_lengths, decState):

    # embed everything
    nt_embeddings = self.nt_embedding(Variable(batch['nt'].cuda(), requires_grad=False))
    rule_embeddings = self.rule_embedding(Variable(batch['prev_rules'].cuda(), requires_grad=False))
    parent_rule_embeddings = self.rule_embedding(Variable(batch['parent_rules'].cuda(), requires_grad=False))

    attn_outputs, attn_scores, copy_attn_scores = [], [], []
    # For each batch we have to maintain states

    batch_size = batch['nt'].size(0) # 1 for predict

    attn_outputs, attn_scores, copy_attn_scores = [], [], []
    for i, (nt, rule, parent_rule) in enumerate(zip(nt_embeddings.split(1, 1), rule_embeddings.split(1, 1), parent_rule_embeddings.split(1, 1))):
      # accumulate parent decoder states
      parent_states = []
      for j in range(0, batch_size):
        try: # this is needed coz the batch is of different sizes
          parent_states.append(batch['parent_states'][j][i]) # one state for every batch
        except:
          parent_states.append(batch['parent_states'][j][0]) # one state for every batch
      parent_states = torch.cat(parent_states, 0)

      rnn_output, prev_hidden = self.decoder_rnn(torch.cat((nt, rule, parent_rule, parent_states), 2), decState.hidden)
      rnn_output.contiguous()
      attn_output, attn_score = self.attn(rnn_output, context, context_lengths)
      attn_scores.append(attn_score)
      attn_output = self.decoder_dropout(attn_output)
      attn_outputs.append(attn_output)
      decState.update_state(prev_hidden, attn_output)
      if self.opt.copy_attn:
        _, copy_attn_score = self.copy_attn(attn_output, context, context_lengths)
        copy_attn_scores.append(copy_attn_score)

      # update all children
      for j, elem in enumerate(rnn_output.split(1, 0)):
        # children wont be there during prediction
        if 'children' in batch and i in batch['children'][j]: # rule i has children
          for child in batch['children'][j][i]:
            batch['parent_states'][j][child] = elem

    output = torch.cat(attn_outputs, 1)
    attn_scores = torch.cat(attn_scores, 1)
    copy_attn_scores = torch.cat(copy_attn_scores, 1) if self.opt.copy_attn else None

    return output, attn_scores, copy_attn_scores

  def predict(self, enc_hidden, context, context_lengths, batch, beam_size, max_code_length, generator, replace_unk, vis_params):
    # This decoder does not have input feeding. Parent state replces that
    decState = DecoderState(
      enc_hidden, #encoder hidden
      Variable(torch.zeros(1, 1, self.opt.rnn_size).cuda(), requires_grad=False) # parent state
    )
    # Repeat everything beam_size times.
    def rvar(a, beam_size):
      return Variable(a.repeat(beam_size, 1, 1), volatile=True)
    context = rvar(context.data, beam_size)
    context_lengths = context_lengths.repeat(beam_size)
    decState.repeat_beam_size_times(beam_size) # TODO: get back to this

    # Use only one beam
    beam = TreeBeam(beam_size, True, self.vocabs, self.opt.rnn_size)

    for count in range(0, max_code_length): # We will break when we have the required number of terminals
      # to be consistent with seq2seq

      if beam.done(): # TODO: fix b.done
        break

      # Construct batch x beam_size nxt words.
      # Get all the pending current beam words and arrange for forward.
      # Uses the start symbol in the beginning
      inp = beam.getCurrentState() # Should return a batch of the frontier

      # Run one step., decState gets automatically updated
      output, attn, copy_attn = self.forward(inp, context, context_lengths, decState)
      scores = generator(bottle(output), bottle(copy_attn), batch['src_map'], inp) #generator needs the non-terminals

      out = generator.collapseCopyScores(unbottle(scores.data.clone(), beam_size), batch) # needs seq2seq from batch
      out = out.log()

      # beam x tgt_vocab
      beam.advance(out[:, 0],  attn.data[:, 0], output)
      decState.beam_update(beam.getCurrentOrigin(), beam_size)

    score, times, k = beam.getFinal() # times is the length of the prediction
    hyp, att = beam.getHyp(times, k)
    goldNl = self.vocabs['seq2seq'].addStartOrEnd(batch['raw_seq2seq'][0]) # because batch = 1
    goldCode = self.vocabs['code'].addStartOrEnd(batch['raw_code'][0])
    # goldProd = self.vocabs['next_rules'].addStartOrEnd(batch['raw_next_rules'][0])
    predSent = self.buildTargetTokens(
      hyp,
      self.vocabs,
      goldNl,
      att,
      batch['seq2seq_vocab'][0],
      replace_unk
    )
    predSent = ProdDecoder.rulesToCode(predSent)
    return Prediction(goldNl, goldCode, predSent, att)


  @staticmethod
  def rulesToCode(rules):
    stack = []
    code = []
    for i in range(0, len(rules)):
      if not CDDataset._is_terminal_rule(rules[i]):
        stack.extend(rhs(rules[i]).split('___')[::-1])
      else:
        code.append(rhs(rules[i]))

      try:
        top = stack.pop()

        while not top[0].isupper():
          code.append(top)
          if len(stack) == 0:
            break
          top = stack.pop()
      except:
        pass

    return code

  def buildTargetTokens(self, pred, vocabs, src, attn, copy_vocab, replace_unk):
      vocab = vocabs['next_rules']
      tokens = []
      for tok in pred:
          if tok < len(vocab):
              tokens.append(vocab.itos[tok])
          else:
              tokens.append("IdentifierNT-->" + copy_vocab.itos[tok - len(vocab)])

      if replace_unk and attn is not None:
          for i in range(len(tokens)):
              if tokens[i] == '<unk>':
                  _, maxIndex = attn[i].max(0)
                  tokens[i] = "IdentifierNT-->" + src[maxIndex[0]]

      return tokens
