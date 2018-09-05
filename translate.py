import argparse
from preprocess import Vocab, CDDataset
import torch
from S2SModel import S2SModel
import sys

parser = argparse.ArgumentParser(description='translate.py')

parser.add_argument('-model', required=True,
                    help='Path to model .pt file')
parser.add_argument('-src',   required=True,
                    help='Source sequence to decode (one line per sequence)')
parser.add_argument('-output', default='pred.txt',
                    help="""Path to output the predictions (each line will
                    be the decoded sequence""")
parser.add_argument('-beam_size',  type=int, default=5,
                    help='Beam size')
parser.add_argument('-batch_size', type=int, default=1,
                    help='Batch size')
parser.add_argument('-max_sent_length', type=int, default=100,
                    help='Maximum sentence length.')
parser.add_argument('-replace_unk', action="store_true",
                    help="""Replace the generated UNK tokens with the source
                    token that had highest attention weight. If phrase_table
                    is provided, it will lookup the identified source token and
                    give the corresponding target token. If it is not provided
                    (or the identified source token does not exist in the
                    table) then it will copy the source token""")
parser.add_argument('-gpu', type=int, default=0,
                    help="Device to run on")
parser.add_argument('-trunc', type=int, default=-1,
                    help="Truncate test set.")

def main():
  opt = parser.parse_args()
  torch.cuda.set_device(opt.gpu)
  checkpoint = torch.load(opt.model, map_location=lambda storage, loc: storage)
  vocabs = checkpoint['vocab']
  vocabs['mask'] = vocabs['mask'].cuda()

  test = CDDataset(opt.src, None, test=True, trunc=opt.trunc)
  test.toNumbers(checkpoint['vocab'])
  total_test = test.compute_batches(opt.batch_size, checkpoint['vocab'], checkpoint['opt'].max_camel, 0, 1, checkpoint['opt'].decoder_type, randomize=False, no_filter=True)
  sys.stderr.write('Total test: {}'.format(total_test))
  sys.stderr.flush()

  model = S2SModel(checkpoint['opt'], vocabs)
  model.load_state_dict(checkpoint['model'])
  model.cuda()
  model.eval()

  predictions = []
  for idx, batch in enumerate(test.batches): # For each batch
    predictions.append(model.predict(batch, opt, None))

  for idx, prediction in enumerate(predictions):
    prediction.output(opt.output, idx)

if __name__ == "__main__":
  main()
