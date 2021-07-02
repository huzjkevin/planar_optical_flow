import os
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter
import tqdm
from torch.optim.lr_scheduler import StepLR, MultiStepLR, ExponentialLR

def checkpoint_state(model=None, optimizer=None, epoch=None, it=None):
    optim_state = optimizer.state_dict() if optimizer is not None else None
    if model is not None:
        if isinstance(model, torch.nn.DataParallel):
            model_state = model.module.state_dict()
        else:
            model_state = model.state_dict()
    else:
        model_state = None

    return {'epoch': epoch, 'it': it, 'model_state': model_state, 'optimizer_state': optim_state}

def save_checkpoint(state=None, filename='checkpoint'):
    filename = '{}.pth'.format(filename)
    torch.save(state, filename)

def load_checkpoint(model=None, optimizer=None, filename='checkpoint', logger=None):
    if os.path.isfile(filename):
        checkpoint = torch.load(filename)
        epoch = checkpoint['epoch'] if 'epoch' in checkpoint.keys() else -1
        it = checkpoint.get('it', 0.0)
        if model is not None and checkpoint['model_state'] is not None:
            model.load_state_dict(checkpoint['model_state'])
        if optimizer is not None and checkpoint['optimizer_state'] is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state'])
    else:
        print('Could not find %s' % filename)
        raise FileNotFoundError

    return it, epoch

def lr_scheduler():
    return 0.01

class LucasScheduler(object):
    """
    Return `v0` until `e` reaches `e0`, then exponentially decay
    to `v1` when `e` reaches `e1` and return `v1` thereafter, until
    reaching `eNone`, after which it returns `None`.

    Copyright (C) 2017 Lucas Beyer - http://lucasb.eyer.be =)
    """
    def __init__(self, optimizer, e0, v0, e1, v1, eNone=float('inf')):
        self.e0, self.v0 = e0, v0
        self.e1, self.v1 = e1, v1
        self.eNone = eNone
        self._optim = optimizer

    def step(self, epoch):
        if epoch < self.e0:
            lr = self.v0
        elif epoch < self.e1:
            lr = self.v0 * (self.v1/self.v0)**((epoch-self.e0)/(self.e1-self.e0))
        elif epoch < self.eNone:
            lr = self.v1

        for group in self._optim.param_groups:
            group['lr'] = lr

    def get_lr(self):
        return self._optim.param_groups[0]['lr']

class Trainer(object):
    def __init__(self, model, model_fn,  optimizer, ckpt_dir, lr_scheduler, model_fn_eval=None, grad_norm_clip=1.0, tb_logger=None, logger=None):
        # Model
        self.model = model
        self.model_fn = model_fn
        self.model_fn_eval = model_fn_eval

        # Training
        self.optimizer = optimizer
        self.ckpt_dir = ckpt_dir
        self.grad_norm_clip = grad_norm_clip
        self.lr_scheduler = lr_scheduler
        self._epoch = 0
        self._it = 0

        # Logging
        self.tb_logger = tb_logger
        self.logger = logger

    def train(self, num_epochs, train_loader, eval_loader=None, eval_frequency=1, ckpt_save_interval=5, lr_scheduler_each_iter=True, starting_epoch=0, starting_iteration=0):
        self._it = starting_iteration

        for self._epoch in range(starting_epoch, num_epochs):
            if not lr_scheduler_each_iter:
                self.lr_scheduler.step(self._epoch)

            running_loss = 0.0

            #Train for one epoch
            for cur_it, batch in enumerate(train_loader):
                if lr_scheduler_each_iter:
                    self.lr_scheduler.step(self._epoch + cur_it / len(train_loader))

                cur_lr = self.lr_scheduler.get_lr()
                self.tb_logger.add_scalar('Learning_rate', cur_lr, self._it)

                #batch loss
                # loss, pred_norm, target_norm = self._train_it(batch)
                loss = self._train_it(batch)

                running_loss += loss
                # avg_pred_norm += pred_norm #Test
                # avg_target_norm += target_norm #Test
                self.tb_logger.add_scalar('Train_loss', loss, self._it)

                self._it += 1

            #save trained model
            trained_epoch = self._epoch + 1
            print("Current Epoch: %d" % trained_epoch, " [Learning rate: {}]".format(cur_lr))
            print("Epoch loss: ", running_loss / len(train_loader))
            # print("Average pred norm: ", avg_pred_norm / len(train_loader)) #Test
            # print("Average target norm: ", avg_target_norm / len(train_loader)) #Test
            # Log to tensorboard
            self.tb_logger.add_scalar('Epoch_loss', running_loss / len(train_loader), self._epoch)

            if trained_epoch % ckpt_save_interval == 0:
                ckpt_name = os.path.join(self.ckpt_dir, "ckpt_e{}".format(trained_epoch))
                print("Saving checkpoint to {}".format(ckpt_name))
                save_checkpoint(checkpoint_state(self.model, self.optimizer, trained_epoch, self._it), filename=ckpt_name)

            # eval one epoch
            if eval_loader is not None and trained_epoch % eval_frequency == 0:
                with torch.set_grad_enabled(False):
                    eval_loss, loss_dim, loss_ori, iou = self.model_fn_eval(self.model, eval_loader)

                    self.logger.info('****************** Epoch Evaluation ******************')
                    self.logger.info('Validation, eval loss: {}'.format(eval_loss))
                    self.logger.info('Validation, dimension loss: {} [m]'.format(loss_dim))
                    self.logger.info('Validation, orientation loss: {} [rad]'.format(loss_ori))
                    self.logger.info('Validation, avg IOU: {}'.format(iou))
                    self.logger.info('********************* Epoch End **********************')

                    self.tb_logger.add_scalar('val_loss', eval_loss, self._epoch)
                    self.tb_logger.add_scalar('val_loss_dimension', loss_dim, self._epoch)
                    self.tb_logger.add_scalar('val_loss_orientation', loss_ori, self._epoch)
                    self.tb_logger.add_scalar('val_avg_IOU', iou, self._epoch)

            self.tb_logger.flush()

    def _train_it(self, batch):
        self.model.train()  #Set the model to training mode
        self.optimizer.zero_grad()  #Clear the gradients before training

        # loss, pred_norm, target_norm = self.model_fn(self.model, batch)
        loss = self.model_fn(self.model, batch)

        loss.backward()
        if self.grad_norm_clip > 0:
            clip_grad_norm_(self.model.parameters(), self.grad_norm_clip)
        self.optimizer.step()

        # return loss.item(), pred_norm.item(), target_norm.item()
        return loss.item()