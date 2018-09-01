from Statistics import Statistics
import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_norm
import torch.nn as nn
import time

class Trainer:
  def __init__(self, model):
    self.model = model
    self.opt = model.module.opt if isinstance(model, nn.parallel.DistributedDataParallel) else model.opt
    self.start_epoch = self.opt.start_epoch if self.opt.start_epoch  else 1

    self.lr = self.opt.learning_rate
    self.betas = [0.9, 0.98]
    self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr,
                                betas=self.betas, eps=1e-9)

    if 'prev_optim' in self.opt:
      print('Loading prev optimizer state')
      self.optimizer.load_state_dict(self.opt.prev_optim)
      for state in self.optimizer.state.values():
        for k, v in state.items():
          if isinstance(v, torch.Tensor):
            state[k] = v.cuda()

  def save_checkpoint(self, epoch, valid_stats):
      real_model = (self.model.module
                    if isinstance(self.model, nn.parallel.DistributedDataParallel)
                    else self.model)

      model_state_dict = real_model.state_dict()
      self.opt.learning_rate = self.lr
      checkpoint = {
          'model': model_state_dict,
          'vocab': real_model.vocabs,
          'opt':   self.opt,
          'epoch': epoch,
          'optim': self.optimizer.state_dict()
        }
      torch.save(checkpoint,
                 '%s_acc_%.2f_ppl_%.2f_e%d.pt'
                 % (self.opt.save_model + '/model', valid_stats.accuracy(),
                    valid_stats.ppl(), epoch))

  def update_learning_rate(self, valid_stats):
    if self.last_ppl is not None and valid_stats.ppl() > self.last_ppl:
        self.lr = self.lr * self.opt.learning_rate_decay
        print("Decaying learning rate to %g" % self.lr)

    self.last_ppl = valid_stats.ppl()
    self.optimizer.param_groups[0]['lr'] = self.lr

  def run_train_batched(self, train_data, valid_data, vocabs):
    print(self.model.parameters)

    total_train = train_data.compute_batches(self.opt.batch_size, vocabs, self.opt.max_camel, 0, 1, self.opt.decoder_type,  trunc=self.opt.trunc)
    total_valid = valid_data.compute_batches(10 if self.opt.decoder_type in ["prod", "concode"] else self.opt.batch_size, vocabs, self.opt.max_camel, 0, 1, self.opt.decoder_type, randomize=False, trunc=self.opt.trunc)

    print('Computed Batches. Total train={}, Total valid={}'.format(total_train, total_valid))

    report_stats = Statistics()
    self.last_ppl = None

    for epoch in range(self.start_epoch, self.opt.epochs + 1):
      self.model.train()

      total_stats = Statistics()
      for idx, batch in enumerate(train_data.batches):
        loss, batch_stats = self.model.forward(batch)
        batch_size = batch['code'].size(0)
        loss.div(batch_size).backward()
        report_stats.update(batch_stats)
        total_stats.update(batch_stats)

        clip_grad_norm(self.model.parameters(), self.opt.max_grad_norm)
        self.optimizer.step()
        self.optimizer.zero_grad()

        if (idx + 1) % self.opt.report_every == -1 % self.opt.report_every:
          report_stats.output(epoch, idx + 1, len(train_data.batches), total_stats.start_time)
          report_stats = Statistics()

      print('Train perplexity: %g' % total_stats.ppl())
      print('Train accuracy: %g' % total_stats.accuracy())

      self.model.eval()
      valid_stats = Statistics()
      for idx, batch in enumerate(valid_data.batches):
        loss, batch_stats = self.model.forward(batch)
        valid_stats.update(batch_stats)

      print('Validation perplexity: %g' % valid_stats.ppl())
      print('Validation accuracy: %g' % valid_stats.accuracy())

      self.update_learning_rate(valid_stats)
      print('Saving model')
      self.save_checkpoint(epoch, valid_stats)
      print('Model saved')
