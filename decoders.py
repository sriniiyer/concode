from torch.autograd import Variable

class DecoderState():
    """ Input feed is ignored for this work"""
    def __init__(self, rnnstate, input_feed):
      self.hidden = rnnstate
      self.input_feed = input_feed
      self.batch_size = rnnstate[0].size(0)
      self.rnn_size = rnnstate[0].size(2)

    def clone(self):
      return DecoderState((self.hidden[0].clone(), self.hidden[1].clone()), self.input_feed.clone() if self.input_feed is not None else None)

    def update_state(self, rnnstate, input_feed):
      self.hidden = rnnstate
      self.input_feed = input_feed

    def repeat_beam_size_times(self, beam_size):
      """ Repeat beam_size times along batch dimension. """
      # Vars contains h, c, and input feed. Separate it later
      self.hidden = [Variable(e.data.repeat(1, beam_size, 1), volatile=True)
                               for e in self.hidden]
      self.input_feed = Variable(self.input_feed.data.repeat(beam_size, 1, 1), volatile=True)

    def beam_update(self, positions, beam_size):
      """ Update when beam advances. """
      for e in self.hidden:
        a, br, d = e.size()
        # split batch x beam into two separate dimensions
        # in order to pick the particular beam that
        # we want to update
        # Choose beam number idx
        e.data.copy_(
            e.data.index_select(1, positions))

      br, a, d = self.input_feed.size() 
      self.input_feed.data.copy_(
          self.input_feed.data.index_select(0, positions))

class Prediction():
  def __init__(self, goldNl, goldCode, prediction, attn):
    self.goldNl = goldNl
    self.goldCode = goldCode
    self.prediction = prediction
    self.attn = attn

  def output(self, prefix, idx):
    out_file = open(prefix, 'a')
    debug_file = open(prefix + '.html', 'a')

    out_file.write(' '.join(self.prediction) + '\n')

    debug_file.write('<b>Id:</b>' + str(idx) + '<br>')
    debug_file.write('<b>Language:</b>' + '<br>')
    debug_file.write(' '.join(self.goldNl) + '<br>')
    debug_file.write('<b>Code:</b>' + '<br>')
    debug_file.write(' '.join(self.goldCode) + '<br>')

    out_file.close()
    debug_file.close()
