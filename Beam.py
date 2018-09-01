import torch
from preprocess import rhs, CDDataset
from torch.autograd import Variable
import copy

class TreeBeam(object):
    def __init__(self, size, cuda, vocabs, rnn_size):
        self.size = size
        self.vocabs = vocabs
        self.tt = torch.cuda if cuda else torch
        self.rnn_size = rnn_size

        # The score for each translation on the beam.
        self.scores = self.tt.FloatTensor(size).zero_()

        # The backpointers at each time-step.
        self.prevKs = []

        # The outputs at each time-step.
        # Start with one element
        self.nextYs = [self.tt.LongTensor(self.size)
                       .fill_(self.vocabs['next_rules'].stoi['<blank>'])]
        self.valid = [[0]]
        self.nextYs[0][0] = self.vocabs['prev_rules'].stoi['<s>']
        # This is ok. The first inp is filled in from the stack.
        # and the nt is decided to be <s> based on len(prevks) == 0

        # Has EOS topped the beam yet.
        self.eosTop = False

        # The attentions (matrix) for each time.
        self.attn = []

        # Time and k pair for finished.
        self.finished = []
        self.stacks = [[('MemberDeclaration', '<s>', Variable(self.tt.FloatTensor(1, 1, self.rnn_size).zero_(), requires_grad=False))] for i in range(0, self.size)] # stacks for non terminals
         #nt, parent_rule, parent_state

    def getCurrentState(self):
        "Get the outputs for the current timestep."
        # We need to return a batch here
        # the batch should contain nt, prev_rule, parent_rule, parent_states
        batch = {
          'nt' : self.tt.LongTensor(self.size, 1),
          'prev_rules': self.tt.LongTensor(self.size , 1),
          'parent_rules': self.tt.LongTensor(self.size, 1),
          'parent_states': {}
        }
        for i in range(0, len(self.nextYs[-1])):
          if len(self.prevKs) == 0: # in the beginning
            rule = '<s>'
          elif self.nextYs[-1][i] >= len(self.vocabs['next_rules']):
            rule = '<unk>'
          else:
            rule = self.vocabs['next_rules'].itos[self.nextYs[-1][i]]

          # if the stack is empty put a placeholder
          if len(self.stacks[i]) == 0:
            (nt, parent_rule, parent_state) = ('MemberDeclaration', '<s>', Variable(self.tt.FloatTensor(1, 1, self.rnn_size).zero_(), requires_grad=False))
          else:
            (nt, parent_rule, parent_state) = self.stacks[i][-1] #.top()

          batch['nt'][i][0] = self.vocabs['nt'].stoi[nt]
          batch['prev_rules'][i][0] = self.vocabs['prev_rules'].stoi[CDDataset.getAnonRule(rule)]
          batch['parent_rules'][i][0] = self.vocabs['prev_rules'].stoi[parent_rule]
          batch['parent_states'][i] = {}
          batch['parent_states'][i][0] = parent_state

        return batch

    def getCurrentOrigin(self):
        "Get the backpointers for the current timestep."
        return self.prevKs[-1]

    def advance(self, wordLk, attnOut, rnn_output):
        """
        Given prob over words for every last beam `wordLk` and attention
        `attnOut`: Compute and update the beam search.

        Parameters:

        * `wordLk`- probs of advancing from the last step (K x words)
        * `attnOut`- attention at the last step

        Returns: True if beam search is complete.
        """
        numWords = wordLk.size(1)

        # Sum the previous scores.
        if len(self.prevKs) > 0:
            beamLk = wordLk + self.scores.unsqueeze(1).expand_as(wordLk)

            # Don't let EOS have children.
            for i in range(self.nextYs[-1].size(0)):
                if len(self.stacks[i]) == 0:
                    beamLk[i] = -1e20
        else:
            beamLk = wordLk[0]
        flatBeamLk = beamLk.view(-1)
        bestScores, bestScoresId = flatBeamLk.topk(self.size, 0, True, True)

        self.scores = bestScores

        # bestScoresId is flattened beam x word array, so calculate which
        # word and beam each score came from
        oldStacks = self.stacks
        self.stacks = [[] for i in range(0, self.size)] # stacks for non terminals

        # bestScoresId is flattened beam x word array, so calculate which
        # word and beam each score came from
        prevK = bestScoresId / numWords
        self.prevKs.append(prevK)
        self.nextYs.append((bestScoresId - prevK * numWords))
        self.attn.append(attnOut.index_select(0, prevK))
        self.stacks = [copy.deepcopy(oldStacks[k]) for k in prevK]
        for i in range(0, self.size):
          currentRule = (bestScoresId[i] - prevK[i] * numWords) 
          # currentRule can be a copy index
          if currentRule >= len(self.vocabs['next_rules']):
            rule = '<unk>'
          else:
            rule = self.vocabs['next_rules'].itos[currentRule] 
          try:
            self.stacks[i].pop() # This rule has been processed. This should not error out
          except:
            # This can error out if there are very few options for the previous rules (rest are -inf) and a stack with 1e-20 is also chosen in topk
            pass

          # If its a terminal rule, we dont needs its parents anymore
          if not CDDataset._is_terminal_rule(rule):
            # in the beginning, MemberDeclaration has only 2 options
            # so the third best in the beam is -inf
            # it should get eliminated later because the score is -inf
            if rule != '<blank>':
              for elem in rhs(rule).split('___')[::-1]:
                if elem[0].isupper():
                  self.stacks[i].append((elem, rule, rnn_output[prevK[i]].unsqueeze(0)))

        for i in range(self.nextYs[-1].size(0)):
            if len(self.stacks[i]) == 0:
                s = self.scores[i]
                self.finished.append((s, len(self.nextYs) - 1, i))

        # End condition is when top-of-beam is EOS and no global score.
        if len(self.stacks[0]) == 0:
            self.eosTop = True


    def done(self):
        return self.eosTop and len(self.finished) >= 1

    def getFinal(self):
      if len(self.finished) == 0:
        self.finished.append((self.scores[0], len(self.nextYs) - 1, 0))
      self.finished.sort(key=lambda a: -a[0])
      return self.finished[0]

    def getHyp(self, timestep, k):
        """
        Walk back to construct the full hypothesis.
        """
        hyp, attn = [], []
        for j in range(len(self.prevKs[:timestep]) - 1, -1, -1):
            hyp.append(self.nextYs[j+1][k])
            attn.append(self.attn[j][k])
            k = self.prevKs[j][k]
        return hyp[::-1], torch.stack(attn[::-1])

class Beam(object):
    def __init__(self, size, cuda, vocab):

        self.size = size
        self.vocab = vocab
        self.tt = torch.cuda if cuda else torch

        # The score for each translation on the beam.
        self.scores = self.tt.FloatTensor(size).zero_()

        # The backpointers at each time-step.
        self.prevKs = []

        # The outputs at each time-step.
        self.nextYs = [self.tt.LongTensor(size)
                       .fill_(self.vocab.stoi['<blank>'])]
        self.nextYs[0][0] = self.vocab.stoi['<s>']

        # Has EOS topped the beam yet.
        self._eos = self.vocab.stoi['</s>']
        self.eosTop = False

        # The attentions (matrix) for each time.
        self.attn = []

        # Time and k pair for finished.
        self.finished = []

    def getCurrentState(self):
        "Get the outputs for the current timestep."
        batch = {
          'code' : self.tt.LongTensor(self.nextYs[-1]).view(-1, 1),
        }

        return batch

    def getCurrentOrigin(self):
        "Get the backpointers for the current timestep."
        return self.prevKs[-1]

    def advance(self, wordLk, attnOut):
        """
        Given prob over words for every last beam `wordLk` and attention
        `attnOut`: Compute and update the beam search.

        Parameters:

        * `wordLk`- probs of advancing from the last step (K x words)
        * `attnOut`- attention at the last step

        Returns: True if beam search is complete.
        """
        numWords = wordLk.size(1)

        # Sum the previous scores.
        if len(self.prevKs) > 0:
            beamLk = wordLk + self.scores.unsqueeze(1).expand_as(wordLk)

            # Don't let EOS have children.
            for i in range(self.nextYs[-1].size(0)):
                if self.nextYs[-1][i] == self._eos:
                    beamLk[i] = -1e20
        else:
            beamLk = wordLk[0]
        flatBeamLk = beamLk.view(-1)
        bestScores, bestScoresId = flatBeamLk.topk(self.size, 0, True, True)

        self.scores = bestScores

        # bestScoresId is flattened beam x word array, so calculate which
        # word and beam each score came from
        prevK = bestScoresId / numWords
        self.prevKs.append(prevK)
        self.nextYs.append((bestScoresId - prevK * numWords))
        self.attn.append(attnOut.index_select(0, prevK))

        for i in range(self.nextYs[-1].size(0)):
            if self.nextYs[-1][i] == self._eos:
                s = self.scores[i]
                self.finished.append((s, len(self.nextYs) - 1, i))

        # End condition is when top-of-beam is EOS and no global score.
        if self.nextYs[-1][0] == self.vocab.stoi['</s>']:
            self.eosTop = True

    def done(self):
        return self.eosTop and len(self.finished) >= 1

    def getFinal(self):
      if len(self.finished) == 0:
        self.finished.append((self.scores[0], len(self.nextYs) - 1, 0))
      self.finished.sort(key=lambda a: -a[0])
      return self.finished[0]

    def getHyp(self, timestep, k):
        """
        Walk back to construct the full hypothesis.
        """
        hyp, attn = [], []
        for j in range(len(self.prevKs[:timestep]) - 1, -1, -1):
            hyp.append(self.nextYs[j+1][k])
            attn.append(self.attn[j][k])
            k = self.prevKs[j][k]
        return hyp[::-1], torch.stack(attn[::-1])
