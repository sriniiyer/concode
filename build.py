import json
import argparse
import collections
from Tree import getProductions
import os

def processNlToks(nlToks):
  return [tok.encode('ascii', 'replace').decode().strip() for tok in nlToks \
           if tok != "-RCB-" and \
           tok != "-LCB-" and \
           tok != "-LSB-" and \
           tok != "-RSB-" and \
           tok != "-LRB-" and \
           tok != "-RRB-" and \
           tok != "@link" and \
           tok != "@code" and \
           tok != "@inheritDoc" and \
           tok.encode('ascii', 'replace').decode().strip() != '']

trainNls = []

def processFiles(fname, prefix, dset, trunc):
  dataset = []

  codeVocab = collections.Counter()
  nlVocab = collections.Counter()

  i = 0
  didnt_parse = 0

  for line in open(fname, 'r'):
    i += 1
    if i % 10000 == 0:
      print(i)

    js = json.loads(line)
    code = js['renamed']

    codeVocab.update(code)

    nlToks = processNlToks(js['nlToks'])
    nlVocab.update(nlToks)

    codeToks = [cTok.encode('ascii', 'replace').decode().replace("\x0C", "").strip() for cTok in code]
    if len(nlToks) == 0 or len(codeToks) == 0:
      continue

    # put placeholder variables and methods
    if len(js["memberVariables"]) == 0:
      js["memberVariables"]["placeHolder"] = "PlaceHolder"
    if len(js["memberFunctions"]) == 0:
      js["memberFunctions"]["placeHolder"] = [['placeholderType']]


    # pull out methods
    methodNames, methodReturns, methodParamNames, methodParamTypes = [], [], [], []
    for methodName in js["memberFunctions"]:
      for methodInstance in js["memberFunctions"][methodName]:
        # Always have a parameter
        methodNames.append(methodName)
        methodReturns.append("None" if methodInstance[0] is None else methodInstance[0]) # The first element is the return type
        if len(methodInstance) == 1:
          methodInstance += ['NoParams noParams']
        methodParamNames.append([methodInstance[p].split()[-1] for p in range(1, len(methodInstance))])
        methodParamTypes.append([' '.join(methodInstance[p].split()[:-1]).replace('final ', '') for p in range(1, len(methodInstance))])

    # Find and annotate class variables
    memberVarNames = [key.split('=')[0].encode('ascii', 'replace').decode() for key, value in js["memberVariables"].items()]
    memberVarTypes = [value.encode('ascii', 'replace').decode() for key, value in js["memberVariables"].items()]

    for t in range(0, len(codeToks)):
      if codeToks[t] in memberVarNames and (codeToks[t - 1] != '.' or (codeToks[t - 1] == '.' and codeToks[t - 2] == "this")):
        codeToks[t] = 'concodeclass_' + codeToks[t]
      elif codeToks[t] == '(' and codeToks[t - 1] in methodNames and (codeToks[t - 2] != '.' or (codeToks[t - 2] == '.' and codeToks[t - 3] == "this")):
        codeToks[t - 1] = 'concodefunc_' + codeToks[t - 1]

    try:
      rule_seq = getProductions('class TestClass { ' + ' '.join(codeToks) + ' }')
    except:
      import pdb
      pdb.set_trace()

    # If it doesnt parse, we should skip this
    if rule_seq is None:
      didnt_parse += 1
      continue

    if dset == "train":
      trainNls.append(' '.join(nlToks))
    elif " ".join(nlToks) in trainNls:
      continue

    try:
      seq2seq = ( ' '.join(nlToks).lower() + ' concode_field_sep ' + \
    ' concode_elem_sep '.join([vtyp + ' ' + vnam for (vnam, vtyp) in zip(memberVarNames, memberVarTypes)]) + ' concode_field_sep ' + \
      ' concode_elem_sep '.join([mret + ' ' + mname + ' concode_func_sep ' + ' concode_func_sep '.join(mpt + ' ' + mpn for (mpt, mpn) in zip(mpts, mpns) )  for (mret, mname, mpts, mpns) in zip(methodReturns, methodNames, methodParamTypes, methodParamNames)] ) ).split()
      seq2seq_nop = ( ' '.join(nlToks).lower() + ' concode_field_sep ' + \
    ' concode_elem_sep '.join([vtyp + ' ' + vnam for (vnam, vtyp) in zip(memberVarNames, memberVarTypes)]) + ' concode_field_sep ' + \
    ' concode_elem_sep '.join([mret + ' ' + mname for (mret, mname) in zip(methodReturns, methodNames)])).split()
    except:
      import pdb
      pdb.set_trace()

    dataset.append(
      {'nl': nlToks,
       'code': codeToks,
       'idx': js['idx'], #str(i),
       'varNames': memberVarNames,
       'varTypes': memberVarTypes,
       'rules': rule_seq,
       'methodNames': methodNames,
       'methodReturns': methodReturns,
       'methodParamNames': methodParamNames,
       'methodParamTypes': methodParamTypes,
       'seq2seq': seq2seq,
       'seq2seq_nop': seq2seq_nop
       })

    if len(dataset) == trunc:
      break

  f = open(prefix + '.dataset', 'w')
  f.write(json.dumps(dataset, indent=4))
  f.close()

  print ('Total code vocab: ' + str(len(codeVocab)))
  print ('Total nl vocab: ' + str(len(nlVocab)))
  print ('Total didnt parse: ' + str(didnt_parse))

if __name__ == '__main__':

  parser = argparse.ArgumentParser(description='build.py')

  parser.add_argument('-train_file', required=True,
                      help="Path to the training source data")
  parser.add_argument('-valid_file', required=True,
                      help="Path to the validation source data")
  parser.add_argument('-test_file', required=True,
                      help="Path to the test source data")
  parser.add_argument('-train_num', type=int, default=100000,
                      help="No. of Training examples")
  parser.add_argument('-valid_num', type=int, default=2000,
                      help="No. of Validation examples")

  parser.add_argument('-output_folder', required=True,
                      help="Output folder for the prepared data")
  opt = parser.parse_args()
  print(opt)

  try:
    os.makedirs(opt.output_folder)
  except:
    pass

  processFiles(opt.train_file, opt.output_folder + '/train', "train", opt.train_num)
  processFiles(opt.valid_file, opt.output_folder + '/valid', "valid", opt.valid_num)
  processFiles(opt.test_file, opt.output_folder + '/test', "test", opt.valid_num)
