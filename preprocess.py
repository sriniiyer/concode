import argparse
import torch
from collections import Counter
import random
import re
import json
import time
import numpy as np

def rhs(rule):
  return rule.split('-->', 1)[1]

def lhs(rule):
  return rule.split('-->', 1)[0]

# parents is a dict that stores each rules' parent 
def getChildrenFromProd(rules, index, node, parent, parents):
  lhs, rhs = rules[index].split('-->')
  parents[index] = parent
  parent = index
  assert (lhs == node)
  for r in rhs.split('___'):
    if lhs == "IdentifierNT" or (not r[0].isupper()) or r == "VarCopy" or r == "MethodCopy": #terminal, ignore it
      pass
    else:
      index = getChildrenFromProd(rules, index + 1, r, parent, parents)
  return index

def isGetter(codeToks):
  return re.search(r"function \( \) \{ return concodeclass_[a-zA-Z0-9_]+ ; \}", ' '.join(codeToks)) != None

def isSetter(codeToks):
  return re.search(r"function \( .* \) \{ concodeclass_[a-zA-Z0-9_]+ = .* ; \}", ' '.join(codeToks)) != None

def split_camel_case(identifier):
  matches = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', identifier)
  return [m.group(0).lower() for m in matches]

def combine_dicts(d1, d2):
  comb = d1
  for k in d2:
    if k not in comb:
      comb[k] = d2[k]
    else:
      for val in d2[k]:
        if val not in comb[k]:
          comb[k].append(val)

  return comb

def expandBatchOneHot(batch, pad, width=None):
  vocab_size = batch.max() + 1 if width == None else width
  new_batch = np.full((batch.size(0), batch.size(1), vocab_size), 0) # This is a mask.
  for i in range(0, batch.size(0)):
    for j in range(0, batch.size(1)):
      if batch.dim() == 3:
        for k in range(0, batch.size(2)):
          if batch[i][j][k] != pad: # pad is 0. Ignore it
            new_batch[i][j][batch[i][j][k]] = 1.0
      elif batch.dim() == 2:
        if batch[i][j] != pad:
          new_batch[i][j][batch[i][j]] = 1.0
  return torch.FloatTensor(new_batch)

def make_batch_elem_into_tensor(batch, entry, pad):
  seq_len = max(len(elem[entry]) for elem in batch)
  torch_batch = np.full((len(batch), seq_len), pad) #torch.LongTensor(seq_len, len(batch)).fill_(pad)
  for i in range(0, len(batch)):
    for j in range(0, len(batch[i][entry])):
      torch_batch[i][j] = batch[i][entry][j]
  return torch.LongTensor(torch_batch)

def make_batch_char_elem_into_tensor(batch, entry, pad, maxl=None, minl=None):
  max_char_length = min(maxl, max(len(chars) for elem in batch for chars in elem[entry]))
  max_char_length = max(max_char_length, minl)
  torch_batch = np.full((len(batch), max_char_length, max(len(elem[entry]) for elem in batch)), pad)

  for i in range(0, len(batch)):
    for j in range(0, len(batch[i][entry])):
      for k in range(0, min(max_char_length, len(batch[i][entry][j]))):
        torch_batch[i][k][j] = batch[i][entry][j][k]
  return torch.LongTensor(torch_batch)

class Vocab():

  def addSymbol(self, sym):
    self.stoi[sym] = len(self.itos)
    self.itos.append(sym)

  def __init__(self, elements, prune, max_num, start=True, stop=True, pad=True, unk=True, rule=False):
    self.start = start
    self.stop = stop
    vocab = Counter()
    self.max_num = max_num
    self.itos = []
    self.stoi = {}
    if pad:
      self.addSymbol('<blank>')
    if unk:
      self.addSymbol('<unk>')
    if start:
      self.addSymbol('<s>')
    if stop:
      self.addSymbol('</s>')

    for w in elements:
      vocab[w] += 1

    for (w, f) in vocab.most_common(self.max_num):
      if (f >  prune or (rule == True and not CDDataset._is_terminal_rule(w))):
        self.itos.append(w)
        self.stoi[w] = len(self.itos) - 1
      else: #map everything else to unk
        self.stoi[w] = self.stoi['<unk>']

  def __len__(self):
    return len(self.itos)

  def addStartOrEnd(self, words):
     return (['<s>'] if self.start else []) + words + (["</s>"] if self.stop else [])

  # the char parameter is only for recursion
  def to_num(self, words, char=0, start=True, stop=True):
      # will be 2 dimensional if its char
      if char > 0:
        ret = [self.to_num(list(word), char=char - 1) for word in words]
      else:
        try:
          ret = ([self.stoi['<s>']] if self.start and start else []) + [self.stoi[w] if w in self.stoi else self.stoi['<unk>'] for w in words] + ([self.stoi["</s>"]] if self.stop and stop else [])
        except:
          import ipdb
          ipdb.set_trace()
      return ret

class Dataset():
  def compute_batches(self, batch_size, vocabs, max_camel, rank, num_gpus, decoder_type, randomize=True, trunc=-1, no_filter=False):
    timer = time.process_time()

    self.batches = []
    curr_batch = []
    total = 0
    for i in range(rank, len(self.examples), num_gpus):
      if not no_filter and decoder_type in ["concode", "prod"] and len(self.examples[i]['next_rules']) > 200:
        continue
      total += 1
      curr_batch.append(self.examples[i])
      if len(curr_batch) == batch_size or i == (len(self.examples) - 1) or i == trunc:
        self.batches.append(self.make_batch_into_tensor(curr_batch, vocabs, max_camel))
        curr_batch = []
      if i == trunc:
        break

    if randomize:
      random.shuffle(self.batches)
    print('Computed batched in :' + str(time.process_time() - timer) + ' secs')
    return total

class CDDataset(Dataset):
  @staticmethod
  def _is_terminal_rule(rule):
    return ("IdentifierNT-->" in rule and rule != 'IdentifierNT-->VarCopy' and rule != 'IdentifierNT-->MethodCopy')\
      or re.match(r"^Nt_.*_literal-->.*", rule) \
      or rule == "<unk>"

  @staticmethod
  def getAnonRule(rule):
    return "Identifier_OR_Literal" if CDDataset._is_terminal_rule(rule) else rule

  def __init__(self, dataFile, opt, test=False, trunc=-1):
    self.examples = []
    self.rhs = {}
    dataset = json.loads(open(dataFile, 'r').read())

    max_code = max(len(js['code']) for js in dataset)
    print('Maximum code toks: ' + str(max_code))
    for js in dataset:
      if test or (len(js['seq2seq']) <= opt.src_seq_length and len(js['code']) <= opt.tgt_seq_length):

        # Important: This should be done after copy!
        for i in range(0, len(js['rules'])):
          js['rules'][i] = js['rules'][i].replace('concodeclass_', '').replace('concodefunc_', '')


        nonTerminals = [rule.split('-->')[0] for rule in js['rules']]
        prevRules = [CDDataset.getAnonRule(x) for x in js['rules']][:-1]

        parents = {}
        children = {}
        parentRules = []
        getChildrenFromProd(js['rules'], 0, "MemberDeclaration", -1, parents)
        for i in range(0, len(js['rules'])):
          if i > 0: # When i == 0, the parent will be <s>, and it will be appended by the vocab[prev_rules]
            parentRules.append(CDDataset.getAnonRule(js['rules'][parents[i]]))

          if parents[i] not in children:
            children[parents[i]] = []
          children[parents[i]].append(i)

        src = [x.lower() for x in js['nl']]
        self.examples.append(
          {'src': src,
           'origcode': js['code'],
           'code': [x.replace('concodeclass_', '').replace('concodefunc_', '') for x in js['code']],
           'varNames': js['varNames'],
           'varTypes': js['varTypes'],
           'methodNames': js['methodNames'],
           'methodReturns': js['methodReturns'],
           'next_rules': js['rules'],
           'prev_rules': prevRules,
           'parent_rules': parentRules,
           'nt': nonTerminals,
           'seq2seq': js["seq2seq_nop"],
           'seq2seq_vocab': Vocab(js['seq2seq_nop'], 0, 100000000, start=False, stop=False), # A vocab just for this sentence
           'children' : children,

           'concode':[j for i in zip(js['varTypes'], js['varNames']) for j in i] + [j for i in zip(js['methodReturns'], js['methodNames']) for j in i],
           'concode_vocab': Vocab(js['varNames'] + js['varTypes']  + js['methodReturns'] + js['methodNames'] + ['concode_copy_placeholder'], 0, 1000000, start=False, stop=False),
           'concode_var': [j for i in zip(js['varTypes'], js['varNames']) for j in i],
           'concode_method': [j for i in zip(js['methodReturns'], js['methodNames']) for j in i],
           }
        )


        #compute seq2seq copy vector
        seq2seq_copy = []
        for w in range(0, len(self.examples[-1]['code'])):
          codeTok = self.examples[-1]['code'][w]
          tmpCopy = []
          for s in range(0, len(self.examples[-1]['seq2seq'])):
            srcTok = self.examples[-1]['seq2seq'][s]
            if srcTok == codeTok and srcTok != ';' and srcTok != ':':
              tmpCopy.append(1)
            else:
              tmpCopy.append(0)
          seq2seq_copy.append(tmpCopy)
        self.examples[-1]['seq2seq_copy'] = seq2seq_copy

        # For every nt, store the list
        # of possible rights
        for rule in js['rules']:
          (nt, r) = rule.split('-->')
          if nt not in self.rhs:
            self.rhs[nt] = []
          if rule not in self.rhs[nt]:
            self.rhs[nt].append(rule)
        if len(self.examples) == trunc: # If trunc is -1, this will never be true
          break

      if len(self.examples) % 100 == 0:
        print("Done: " + str(len(self.examples)))

    # sort by src length
    if not test:
      self.examples.sort(key=lambda x: len(x['src']), reverse=True)

  def toNumbers(self, vocabs):
    for e in self.examples:

      e['seq2seq_nums'] = vocabs['seq2seq'].to_num(e['seq2seq'])
      e['code_nums'] = vocabs['code'].to_num(e['code'])
      e['seq2seq_in_src_nums'] = e['seq2seq_vocab'].to_num(vocabs['seq2seq'].addStartOrEnd(e['seq2seq'])) # use the local vocab for this sentence
      e['code_in_src_nums'] = e['seq2seq_vocab'].to_num(vocabs['code'].addStartOrEnd(e['code'])) # use the local vocab for this sentence
      # For concode decoder--------------
      # We have to do this because we concat them in the decoder
      # and there is padding between the nl, vars and methods in the same example because of batching
      e['src_in_src_nums'] = e['concode_vocab'].to_num(e['src']) # use the local vocab for this sentence
      e['var_in_src_nums'] = e['concode_vocab'].to_num(e['concode_var']) # use the local vocab for this sentence
      e['method_in_src_nums'] = e['concode_vocab'].to_num(e['concode_method']) # use the local vocab for this sentence
      #-------------------------------------------------------
      e['concode_next_rules_in_src_nums'] = e['concode_vocab'].to_num(
        vocabs['next_rules'].addStartOrEnd(
          [rhs(x) if lhs(x) == "IdentifierNT" else '<unk>' for x in e['next_rules']]
        )) # use the local vocab for this sentence
      #------------------------

      e['next_rules_in_src_nums'] = e['seq2seq_vocab'].to_num(
        vocabs['next_rules'].addStartOrEnd(
          [rhs(x) if lhs(x) == "IdentifierNT" else '<unk>' for x in e['next_rules']]
        )) # use the local vocab for this sentence

      # ------- Rule decoder
      # There is no unk in the vocab, so this will throw an error
      # if the rule isnt there in the vocab
      e['prev_rules_nums'] = vocabs['prev_rules'].to_num(e['prev_rules'])
      e['parent_rules_nums'] = vocabs['prev_rules'].to_num(e['parent_rules'])

      # We need to ensure that only certain rules can be unked, not all. This
      # is taken care of when building the vocab
      e['nt_nums'] = vocabs['nt'].to_num(e['nt'])
      e['next_rules_nums'] = vocabs['next_rules'].to_num(e['next_rules'])
      #-------------------------------------

      # --- Our Model -----------
      e['src_nums'] = vocabs['names_combined'].to_num(e['src'])
      e['varTypes_nums'] = vocabs['types'].to_num(e['varTypes'])
      e['varNames_nums'] = vocabs['names_combined'].to_num([(split_camel_case(w)) for w in e['varNames']], char=1)
      e['methodNames_nums'] = vocabs['names_combined'].to_num([ (split_camel_case(w)) for w in e['methodNames']], char=1)
      e['methodReturns_nums'] = vocabs['types'].to_num(e['methodReturns'])
      #-----------------------------------

  def outputStats(self, vocabs):
    print('Average NL length: ' + str(sum([len(e['src']) for e in self.examples]) * 1.0 / len(self.examples)))
    print('Average Code Characters: ' + str(sum([len(' '.join(e['code'])) for e in self.examples]) * 1.0 / len(self.examples)))
    print('Average Code Tokens : ' + str(sum([len(e['code']) for e in self.examples]) * 1.0 / len(self.examples)))
    print('Max Code Tokens : ' + str(max([len(e['code']) for e in self.examples])))
    print('Average AST Nodes: ' + str(sum([len(rhs(r).split('___')) for e in self.examples for r in e['next_rules']]) * 1.0 / len(self.examples)))
    print('Max AST Nodes: ' + str(max([len(rhs(r).split('___')) for e in self.examples for r in e['next_rules']]) ))
    print('Percent getters: ' + str(sum([int(isGetter(e['origcode'])) for e in self.examples]) * 1.0 / len(self.examples)))
    print('Percent setters: ' + str(sum([int(isSetter(e['origcode'])) for e in self.examples]) * 1.0 / len(self.examples)))

    var_copies = np.mean([1 if "concodeclass_" in ' '.join(e['origcode']) else 0 for e in self.examples]) * 100.0
    fn_copies = np.mean([1 if "concodefunc_" in ' '.join(e['origcode']) else 0 for e in self.examples]) * 100.0
    def match_source(src, code, names):
      for w in src:
        if  (w not in vocabs['code'].stoi or vocabs['code'].stoi[w] == vocabs['code'].stoi['<unk>']) and w in code and w not in names:
          return True
      return False
    src_copies = np.mean([1 if match_source(e['src'], e['origcode'], e['varNames'] + e['varTypes'] + e['methodReturns'] + e['methodNames']) else 0 for e in self.examples]) * 100.0

    def match_type(type_list, code):
      for typ in type_list:
        if typ in code and (typ not in vocabs['code'].stoi or vocabs['code'].stoi[typ] == vocabs['code'].stoi['<unk>']):
          return True
      return False

    type_copies = np.mean([1 if match_type(e['methodReturns'] + e['varTypes'], e['origcode']) else 0 for e in self.examples]) * 100.0

    print('Number of variable copies: {}, function copies: {}, source copies: {}, Type copies: {} '.format(var_copies, fn_copies, src_copies, type_copies))


  @staticmethod
  def make_batch_into_tensor(batch, vocabs, max_camel):

    torch_batch = {}
    # -------- for seq2seq
    torch_batch['seq2seq'] = make_batch_elem_into_tensor(batch, 'seq2seq_nums', vocabs['seq2seq'].stoi['<blank>'])
    torch_batch['code'] = make_batch_elem_into_tensor(batch, 'code_nums', vocabs['code'].stoi['<blank>'])
    local_vocab_blank = batch[0]['seq2seq_vocab'].stoi['<blank>']
    torch_batch['seq2seq_in_src'] = make_batch_elem_into_tensor(batch, 'seq2seq_in_src_nums', local_vocab_blank)
    # src_map maps positions in the source to source vocab entries, so that we can accumulate copy scores for each vocab entry based on all
    # positions in which it appears
    torch_batch['src_map'] = expandBatchOneHot(torch_batch['seq2seq_in_src'], local_vocab_blank) # src token mapped to vocab

    #-----------for concode
    max_local_vocab_in_batch = max(len(x['concode_vocab']) for x in batch)
    torch_batch['src_in_src'] = make_batch_elem_into_tensor(batch, 'src_in_src_nums', batch[0]['concode_vocab'].stoi['<blank>'])
    torch_batch['var_in_src'] = make_batch_elem_into_tensor(batch, 'var_in_src_nums', batch[0]['concode_vocab'].stoi['<blank>'])
    torch_batch['method_in_src'] = make_batch_elem_into_tensor(batch, 'method_in_src_nums', batch[0]['concode_vocab'].stoi['<blank>'])
    torch_batch['concode_src_map_methods'] = expandBatchOneHot(torch_batch['method_in_src'], batch[0]['concode_vocab'].stoi['<blank>'], width=max_local_vocab_in_batch)
    torch_batch['concode_src_map_vars'] = expandBatchOneHot(torch_batch['var_in_src'], batch[0]['concode_vocab'].stoi['<blank>'], width=max_local_vocab_in_batch)
    torch_batch['concode_vocab'] = [b['concode_vocab'] for b in batch] # Store this for replace unk
    torch_batch['concode_next_rules_in_src_nums'] = make_batch_elem_into_tensor(batch, 'concode_next_rules_in_src_nums', local_vocab_blank)
    torch_batch['concode'] = [b['concode'] for b in batch] # Store this for replace unk
    torch_batch['concode_var'] = [b['concode_var'] for b in batch] # Store this for replace unk
    torch_batch['concode_method'] = [b['concode_method'] for b in batch] # Store this for replace unk
    #---------------------------------------------
    torch_batch['code_in_src_nums'] = make_batch_elem_into_tensor(batch, 'code_in_src_nums', local_vocab_blank)
    torch_batch['next_rules_in_src_nums'] = make_batch_elem_into_tensor(batch, 'next_rules_in_src_nums', local_vocab_blank)
    torch_batch['seq2seq_vocab'] = [b['seq2seq_vocab'] for b in batch] # Store this for replace unk
    torch_batch['raw_code'] = [b['code'] for b in batch] # Store this for replace unk
    torch_batch['raw_seq2seq'] = [b['seq2seq'] for b in batch] # Store this for replace unk
    #-------------------------Prod Decoder
    torch_batch['nt'] = make_batch_elem_into_tensor(batch, 'nt_nums', vocabs['nt'].stoi['<blank>'])
    torch_batch['prev_rules'] = make_batch_elem_into_tensor(batch, 'prev_rules_nums', vocabs['prev_rules'].stoi['<blank>'])
    torch_batch['parent_rules'] = make_batch_elem_into_tensor(batch, 'parent_rules_nums', vocabs['prev_rules'].stoi['<blank>'])

    torch_batch['next_rules'] = make_batch_elem_into_tensor(batch, 'next_rules_nums', vocabs['next_rules'].stoi['<blank>'])
    torch_batch['seq2seq_copy'] = CDDataset.stack_with_padding([torch.LongTensor(b['seq2seq_copy']) for b in batch], 0, start_symbol=True, stop_symbol=True)
    torch_batch['children'] = [b['children'] for b in batch] # Store this for replace unk
    #------------------------------

    #---- Our Encoder --------------
    torch_batch['src'] = make_batch_elem_into_tensor(batch, 'src_nums', vocabs['names_combined'].stoi['<blank>'])
    torch_batch['varTypes'] = make_batch_elem_into_tensor(batch, 'varTypes_nums', vocabs['types'].stoi['<blank>'])
    torch_batch['methodReturns'] = make_batch_elem_into_tensor(batch, 'methodReturns_nums', vocabs['types'].stoi['<blank>'])
    torch_batch['varNames'] = make_batch_char_elem_into_tensor(batch, 'varNames_nums', pad=vocabs['names_combined'].stoi['<blank>'], maxl=max_camel, minl=1)
    torch_batch['methodNames'] = make_batch_char_elem_into_tensor(batch, 'methodNames_nums', pad=vocabs['names_combined'].stoi['<blank>'], maxl=max_camel, minl=1)
    torch_batch['raw_src'] = [b['src'] for b in batch] # Store this for replace unk
    torch_batch['raw_varNames'] = [b['varNames'] for b in batch] # Store this for replace unk
    torch_batch['raw_methodNames'] = [b['methodNames'] for b in batch] # Store this for replace unk
    #-------------------------------------

    return torch_batch

  @staticmethod
  def stack_with_padding(batch, pad_, start_symbol=False, stop_symbol=False):
    max_sizes = [len(batch[0]), len(batch[0][0])]
    for b in batch:
      if len(b) > max_sizes[0]:
        max_sizes[0] = len(b)
      if (len(b[0]) > max_sizes[1]):
        max_sizes[1] = len(b[0])

    t = torch.LongTensor(len(batch), max_sizes[0], max_sizes[1]).fill_(pad_)
    for i in range(0, len(batch)):
      for j in range(0, batch[i].size(0)):
        for k in range(0, batch[i].size(1)):
          t[i][j][k] = batch[i][j][k]

    if start_symbol:
      t = torch.cat((torch.LongTensor(len(batch), 1, max_sizes[1]).fill_(pad_), t), 1)
    if stop_symbol:
      t = torch.cat((t, torch.LongTensor(len(batch), 1, max_sizes[1]).fill_(pad_)), 1)
    return t

  @staticmethod
  def compute_masks(rhs, vocabs):
    masks = torch.LongTensor(len(vocabs['nt'].itos), len(vocabs['next_rules'].itos)).fill_(-10000000)  # nt x rules
    for (nt, rules) in rhs.items():
      nt_num = vocabs['nt'].stoi[nt]
      for r in rules:
        r_num = None
        if r in vocabs['next_rules'].stoi:
          r_num = vocabs['next_rules'].stoi[r]
        elif CDDataset._is_terminal_rule(r):
          r_num = vocabs['next_rules'].stoi['<unk>']
        masks[nt_num][r_num] = 0
    return masks


if __name__ == "__main__":

  parser = argparse.ArgumentParser(description='preprocess.py')

  parser.add_argument('-train', required=True,
                      help="Path to the training source data")
  parser.add_argument('-valid', required=True,
                      help="Path to the validation source data")
  parser.add_argument('-src_seq_length', type=int, default=200,
                      help="Maximum source sequence length")
  parser.add_argument('-tgt_seq_length', type=int, default=150,
                      help="Maximum target sequence length to keep.")
  parser.add_argument('-seq2seq_words_min_frequency', type=int, default=6)
  parser.add_argument('-tgt_words_min_frequency', type=int, default=2)
  parser.add_argument('-names_min_frequency', type=int, default=7)
  parser.add_argument('-train_max', type=int, default=200000)
  parser.add_argument('-valid_max', type=int, default=5000)
  parser.add_argument('-save_data', required=True,
                      help="Output file for the prepared data")
  opt = parser.parse_args()
  print(opt)

  train = CDDataset(opt.train, opt, trunc=opt.train_max)

  print("Building Vocab...")
  vocabs = {

    'names_combined': Vocab(
      [w for e in train.examples for w in e['src']] +\
      [c for e in train.examples for w in e['methodNames'] for c in split_camel_case(w)] + \
      [c for e in train.examples for w in e['varNames'] for c in split_camel_case(w)] +\
      [w for e in train.examples for w in e['varNames']] +\
      [w for e in train.examples for w in e['methodNames']]
      , opt.names_min_frequency, 10000000, start=False, stop=False),

    'types': Vocab(
          [w for e in train.examples for w in e['varTypes']] \
        + [w for e in train.examples for w in e['methodReturns']],
      opt.tgt_words_min_frequency,
      10000000,
      start=False,
      stop=False),

    'nt': Vocab([w for e in train.examples for w in e['nt']], 0, 10000, start=False, stop=False, pad=True, unk=False),
    'seq2seq': Vocab([w for e in train.examples for w in e['seq2seq']], opt.seq2seq_words_min_frequency, 45000, start=False, stop=False),
    'code': Vocab([w for e in train.examples for w in e['code']], opt.tgt_words_min_frequency, 25000),
            }

  valid = CDDataset(opt.valid, opt, trunc=opt.valid_max)

  vocabs['next_rules'] = Vocab(
    [w for e in train.examples for w in e['next_rules']] + \
    [w for e in valid.examples for w in e['next_rules'] if not CDDataset._is_terminal_rule(w)], 
    opt.tgt_words_min_frequency, 10000000, start=False, stop=False, pad=True, rule=True)
  vocabs['prev_rules'] = Vocab(
    [w for e in train.examples for w in e['prev_rules']] + \
    [w for e in valid.examples for w in e['prev_rules']],
    0, 10000000, stop=False, pad=True, unk=False)

  train.toNumbers(vocabs)
  print('Training stats')
  train.outputStats(vocabs)

  print("Building Valid...")
  valid.toNumbers(vocabs)
  print('Valid stats')
  valid.outputStats(vocabs)

  vocabs['rhs'] = combine_dicts(train.rhs, valid.rhs)
  mask = CDDataset.compute_masks(vocabs['rhs'], vocabs) # compute_masks needs rhs
  vocabs['mask'] = mask

  print("Saving train/valid/vocabs")
  print('Vocab Statistics')
  for key in vocabs:
    try:
      print(key + ' : ' + str(len(vocabs[key].itos)) + '/' + str(len(vocabs[key].stoi)) )
    except:
      pass

  torch.save(vocabs, open(opt.save_data + '.vocab.pt', 'wb'))
  torch.save(train, open(opt.save_data + '.train.pt', 'wb'))
  torch.save(valid, open(opt.save_data + '.valid.pt', 'wb'))
