import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
from UtilClass import shiftLeft, bottle, unbottle

def aeq(*args):
    """
    Assert all arguments have the same value
    """
    arguments = (arg for arg in args)
    first = next(arguments)
    assert all(arg == first for arg in arguments), \
        "Not all arguments have the same value: " + str(args)

class RegularGenerator(nn.Module):
  def __init__(self, rnn_size, vocabs, opt):
    super(RegularGenerator, self).__init__()
    self.opt = opt
    self.linear = nn.Linear(rnn_size, len(vocabs['code']))
    self.lsm = nn.LogSoftmax(dim=1)
    self.vocabs = vocabs
    self.target_padding_idx = vocabs['code'].stoi['<blank>']

    weight = torch.ones(len(vocabs['code']))
    weight[vocabs['code'].stoi['<blank>']] = 0
    self.criterion = nn.NLLLoss(weight, size_average=False)

  def forward(self, hidden, attn, src_map, batch):
    return self.lsm(self.linear(hidden))

  def computeLoss(self, scores, batch):
    target = Variable(shiftLeft(batch['code'].cuda(), self.vocabs['code'].stoi['<blank>']).view(-1), requires_grad=False)
    loss = self.criterion(scores, target)
    scores_data = scores.data.clone()
    target_data = target.data.clone() #computeLoss populates this

    pred = scores_data.max(1)[1]
    non_padding = target_data.ne(self.target_padding_idx)
    num_correct = pred.eq(target_data).masked_select(non_padding).sum()
    return loss, non_padding.sum(), num_correct

class ProdGenerator(nn.Module):
  def __init__(self, rnn_size, vocabs, opt):
    super(ProdGenerator, self).__init__()
    self.opt = opt
    self.mask = Variable(vocabs['mask'].float().cuda(), requires_grad=False)
    self.linear = nn.Linear(rnn_size , len(vocabs['next_rules']))  # only non unk rules
    self.linear_copy = nn.Linear(rnn_size, 1)
    self.tgt_pad = vocabs['next_rules'].stoi['<blank>']
    self.tgt_unk = vocabs['next_rules'].stoi['<unk>']
    self.vocabs = vocabs

  def forward(self, hidden, attn, src_map, batch):
    out = self.linear(hidden)
    # batch['nt'] contains padding. 
    batch_by_tlen_, slen = attn.size()
    batch_size, slen_, cvocab = src_map.size()

    non_terminals = batch['nt'].contiguous().cuda().view(-1)
    masked_out = torch.add(out, torch.index_select(self.mask, 0, Variable(non_terminals, requires_grad=False)))
    prob = F.softmax(masked_out, dim=1)

    # Probability of copying p(z=1) batch.
    copy = F.sigmoid(self.linear_copy(hidden))

    # Probibility of not copying: p_{word}(w) * (1 - p(z))
    masked_copy = Variable(non_terminals.cuda().view(-1, 1).eq(self.vocabs['nt'].stoi['IdentifierNT']).float()) * copy
    out_prob = torch.mul(prob,  1 - masked_copy.expand_as(prob)) # The ones without IdentifierNT are left untouched
    mul_attn = torch.mul(attn, masked_copy.expand_as(attn))
    copy_prob = torch.bmm(mul_attn.view(batch_size, -1, slen), Variable(src_map.cuda(), requires_grad=False))
    copy_prob = copy_prob.view(-1, cvocab) # bottle it again to get batch_by_len times cvocab
    return torch.cat([out_prob, copy_prob], 1) # batch_by_tlen x (out_vocab + cvocab)

  def computeLoss(self, scores, batch):

    batch_size = batch['seq2seq'].size(0)

    target = Variable(batch['next_rules'].contiguous().cuda().view(-1), requires_grad=False)
    if self.opt.decoder_type == "prod":
      align = Variable(batch['next_rules_in_src_nums'].contiguous().cuda().view(-1), requires_grad=False)
      align_unk = batch['seq2seq_vocab'][0].stoi['<unk>']
    elif self.opt.decoder_type in ["concode"]:
      align = Variable(batch['concode_next_rules_in_src_nums'].contiguous().cuda().view(-1), requires_grad=False)
      align_unk = batch['concode_vocab'][0].stoi['<unk>']

    offset = len(self.vocabs['next_rules'])

    out = scores.gather(1, align.view(-1, 1) + offset).view(-1).mul(align.ne(align_unk).float()) # all where copy is not unk
    tmp = scores.gather(1, target.view(-1, 1)).view(-1)

    out = out + 1e-20 + tmp.mul(target.ne(self.tgt_unk).float()) + \
                  tmp.mul(align.eq(align_unk).float()).mul(target.eq(self.tgt_unk).float()) # copy and target are unks

        # Drop padding.
    loss = -out.log().mul(target.ne(self.tgt_pad).float()).sum()
    scores_data = scores.data.clone()
    target_data = target.data.clone() #computeLoss populates this

    scores_data = self.collapseCopyScores(unbottle(scores_data, batch_size), batch)
    scores_data = bottle(scores_data)

    # Correct target copy token instead of <unk>
    # tgt[i] = align[i] + len(tgt_vocab)
    # for i such that tgt[i] == 0 and align[i] != 0
    # when target is <unk> but can be copied, make sure we get the copy index right
    correct_mask = target_data.eq(self.tgt_unk) * align.data.ne(align_unk)
    correct_copy = (align.data + offset) * correct_mask.long()
    target_data = (target_data * (1 - correct_mask).long()) + correct_copy

    pred = scores_data.max(1)[1]
    non_padding = target_data.ne(self.tgt_pad)
    num_correct = pred.eq(target_data).masked_select(non_padding).sum()

    return loss, non_padding.sum(), num_correct #, stats

  def collapseCopyScores(self, scores, batch):
    """
    Given scores from an expanded dictionary
    corresponding to a batch, sums together copies,
    with a dictionary word when it is ambigious.
    """
    tgt_vocab = self.vocabs['next_rules']
    offset = len(tgt_vocab)
    for b in range(batch['seq2seq'].size(0)):
      if self.opt.decoder_type == "prod":
        src_vocab = batch['seq2seq_vocab'][b]
      elif self.opt.decoder_type in ["concode"]:
        src_vocab = batch['concode_vocab'][b]

      for i in range(1, len(src_vocab)):
        sw = "IdentifierNT-->" + src_vocab.itos[i]
        ti = tgt_vocab.stoi[sw] if sw in tgt_vocab.stoi else self.tgt_unk
        if ti != self.tgt_unk:
          scores[b, :, ti] += scores[b, :, offset + i]
          scores[b, :, offset + i].fill_(1e-20)
    return scores

class CopyGenerator(nn.Module):
    """
    Generator module that additionally considers copying
    words directly from the source.
    """
    def __init__(self, rnn_size, vocabs, opt):
        super(CopyGenerator, self).__init__()
        self.opt = opt
        self.tgt_dict_size = len(vocabs['code'])
        self.tgt_padding_idx = vocabs['code'].stoi['<blank>']
        self.tgt_unk_idx = vocabs['code'].stoi['<unk>']
        self.vocabs = vocabs
        self.linear = nn.Linear(rnn_size, self.tgt_dict_size)
        self.linear_copy = nn.Linear(rnn_size, 1)
        force_copy=False
        self.criterion = CopyGeneratorCriterion(self.tgt_dict_size, force_copy, self.tgt_padding_idx, self.tgt_unk_idx)

    def forward(self, hidden, copy_attn, src_map, batch):
        """
        Computes p(w) = p(z=1) p_{copy}(w|z=0)  +  p(z=0) * p_{softmax}(w|z=0)
        """
        # CHECKS
        batch_by_tlen, _ = hidden.size()
        batch_by_tlen_, slen = copy_attn.size()
        batch_size, slen_, cvocab = src_map.size()
        aeq(batch_by_tlen, batch_by_tlen_)
        aeq(slen, slen_)

        # Original probabilities.
        logits = self.linear(hidden)
        logits[:, self.tgt_padding_idx] = -float('inf')
        prob = F.softmax(logits, dim=1)

        # Probability of copying p(z=1) batch.
        copy = F.sigmoid(self.linear_copy(hidden))

        # Probibility of not copying: p_{word}(w) * (1 - p(z))
        out_prob = torch.mul(prob,  1 - copy.expand_as(prob))
        mul_attn = torch.mul(copy_attn, copy.expand_as(copy_attn))
        copy_prob = torch.bmm(mul_attn.view(batch_size, -1, slen), Variable(src_map.cuda(), requires_grad=False))
        copy_prob = copy_prob.view(-1, cvocab) # bottle it again to get batch_by_len times cvocab
        return torch.cat([out_prob, copy_prob], 1) # batch_by_tlen x (out_vocab + cvocab)

    def computeLoss(self, scores, batch):
        """
        Args:
            batch: the current batch.
            target: the validate target to compare output with.
            align: the align info.
        """
        batch_size = batch['seq2seq'].size(0)

        self.target = Variable(shiftLeft(batch['code'].cuda(), self.tgt_padding_idx).view(-1), requires_grad=False)

        align = Variable(shiftLeft(batch['code_in_src_nums'].cuda(),  self.vocabs['seq2seq'].stoi['<blank>']).view(-1), requires_grad=False)
        # All individual vocabs have the same unk index
        align_unk = batch['seq2seq_vocab'][0].stoi['<unk>']
        loss = self.criterion(scores, self.target, align, align_unk)

        scores_data = scores.data.clone()
        target_data = self.target.data.clone() #computeLoss populates this

        if self.opt.copy_attn:
          scores_data = self.collapseCopyScores(unbottle(scores_data, batch_size), batch)
          scores_data = bottle(scores_data)

          # Correct target copy token instead of <unk>
          # tgt[i] = align[i] + len(tgt_vocab)
          # for i such that tgt[i] == 0 and align[i] != 0
          # when target is <unk> but can be copied, make sure we get the copy index right
          correct_mask = target_data.eq(self.tgt_unk_idx) * align.data.ne(align_unk)
          correct_copy = (align.data + self.tgt_dict_size) * correct_mask.long()
          target_data = (target_data * (1 - correct_mask).long()) + correct_copy


        pred = scores_data.max(1)[1]
        non_padding = target_data.ne(self.tgt_padding_idx)
        num_correct = pred.eq(target_data).masked_select(non_padding).sum()

        return loss, non_padding.sum(), num_correct #, stats

    def collapseCopyScores(self, scores, batch):
      """
      Given scores from an expanded dictionary
      corresponding to a batch, sums together copies,
      with a dictionary word when it is ambigious.
      """
      tgt_vocab = self.vocabs['code']
      offset = len(tgt_vocab)
      for b in range(batch['seq2seq'].size(0)):
        src_vocab = batch['seq2seq_vocab'][b]
        for i in range(1, len(src_vocab)):
          sw = src_vocab.itos[i]
          ti = tgt_vocab.stoi[sw] if sw in tgt_vocab.stoi else self.tgt_unk_idx
          if ti != self.tgt_unk_idx:
            scores[b, :, ti] += scores[b, :, offset + i]
            scores[b, :, offset + i].fill_(1e-20)
      return scores


class CopyGeneratorCriterion(object):
    def __init__(self, vocab_size, force_copy, tgt_pad, tgt_unk, eps=1e-20):
        self.force_copy = force_copy
        self.eps = eps
        self.offset = vocab_size
        self.tgt_pad = tgt_pad
        self.tgt_unk = tgt_unk

    def __call__(self, scores, target, align, copy_unk):
        # Copy prob.
        out = scores.gather(1, align.view(-1, 1) + self.offset) \
                    .view(-1).mul(align.ne(copy_unk).float())
        tmp = scores.gather(1, target.view(-1, 1)).view(-1)

        # Regular prob (no unks and unks that can't be copied)
        if not self.force_copy:
            # first one = target is not unk
            out = out + self.eps + tmp.mul(target.ne(self.tgt_unk).float()) + \
                  tmp.mul(align.eq(copy_unk).float()).mul(target.eq(self.tgt_unk).float()) # copy and target are unks
        else:
            # Forced copy.
            out = out + self.eps + tmp.mul(align.eq(0).float())

        # Drop padding.
        loss = -out.log().mul(target.ne(self.tgt_pad).float()).sum()
        return loss
