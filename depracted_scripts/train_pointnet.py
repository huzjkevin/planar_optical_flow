# sys.path.append("/home/kevin/HIWI/CV/planar_optical_flow")
# sys.path.append("/home/hu/Projects/planar_optical_flow")
# print(sys.path)

import os
import argparse
from shutil import copyfile
import yaml
import torch
from torch import optim

from src.utils.train_utils import Trainer, load_checkpoint, LucasScheduler
from src.utils.eval_utils import model_fn_Bb_regression, model_fn_eval_box_reg, eval_Bb_regression

from src.utils.log_utils import create_logger, create_tb_logger
from src.depracted.model import BoundingBoxRegressor
# from src.data_handle.depracted.jrdb_dataset import JRDBDataset
from src.data_handle.jrdb_dataset import JRDBPointNet
from torch.utils.data import DataLoader

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(description="arg parser")
parser.add_argument("--cfg", type=str, required=True, help="configuration of the experiment")
parser.add_argument("--ckpt", type=str, required=False, default=None)
args = parser.parse_args()

with open(args.cfg, 'r') as f:
    cfg = yaml.safe_load(f)
    cfg['name'] = os.path.basename(args.cfg).split(".")[0] + cfg['tag']
    print("Using configuration {}".format(cfg['name']))

if __name__ == '__main__':
    root_result_dir = os.path.join('../', 'output_pointnet', cfg['name'])
    os.makedirs(root_result_dir, exist_ok=True)
    copyfile(args.cfg, os.path.join(root_result_dir, os.path.basename(args.cfg)))

    ckpt_dir = os.path.join(root_result_dir, 'ckpts')
    os.makedirs(ckpt_dir, exist_ok=True)

    logger, tb_logger = create_logger(root_result_dir), create_tb_logger(root_result_dir)
    logger.info('**********************Start logging**********************')
    # log to file
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)

    logger.info('Prepare data')
    train_set = JRDBPointNet(split='train', cfg=cfg)
    train_loader = DataLoader(train_set, batch_size=cfg['batch_size'], pin_memory=True,
                              num_workers=cfg['num_workers'], shuffle=True,
                              collate_fn=train_set.collate_batch)

    eval_set = JRDBPointNet(split='val', cfg=cfg, validation=True)
    eval_loader = DataLoader(train_set, batch_size=cfg['batch_size'], pin_memory=True,
                              num_workers=cfg['num_workers'], shuffle=True,
                              collate_fn=train_set.collate_batch)

    # val_set = JRDBPointNet(split='train')
    # val_loader = DataLoader(val_set, batch_size=cfg['batch_size'], pin_memory=True,
    #                           num_workers=cfg['num_workers'], shuffle=True,
    #                           collate_fn=train_set.collate_batch)


    logger.info('Prepare model')
    model = BoundingBoxRegressor(input_dim=2, target_dim=3)
    model.cuda()

    # Prepare training
    logger.info('Prepare training')
    # optimizer = optim.Adam(model.parameters(), lr=tu.lr_scheduler())
    optimizer = optim.Adam(model.parameters(), amsgrad=True)

    lr_scheduler = LucasScheduler(optimizer, 0, 1e-1, 200, 1e-5)

    # Logging
    tb_logger = create_tb_logger(root_result_dir)

    if args.ckpt is not None:
        starting_iteration, starting_epoch = load_checkpoint(model=model,
                                                             optimizer=optimizer,
                                                             filename=args.ckpt,
                                                             logger=logger)
    elif os.path.isfile(os.path.join(ckpt_dir, 'sigterm_ckpt.pth')):
        starting_iteration, starting_epoch = load_checkpoint(model=model,
                                                             optimizer=optimizer,
                                                             filename=os.path.join(ckpt_dir, 'sigterm_ckpt.pth'),
                                                             logger=logger)
    else:
        starting_iteration, starting_epoch = 0, 0

    logger.info('Start training')
    trainer = Trainer(model,
                      model_fn_Bb_regression,
                      optimizer,
                      model_fn_eval=model_fn_eval_box_reg,
                      ckpt_dir=ckpt_dir,
                      lr_scheduler=lr_scheduler,
                      grad_norm_clip=cfg['grad_norm_clip'],
                      tb_logger=tb_logger,
                      logger=logger)

    trainer.train(num_epochs=cfg['epochs'],
                  train_loader=train_loader,
                  eval_loader=eval_loader,
                  # eval_frequency=max(int(cfg['epochs'] / 20), 1),
                  eval_frequency=2,
                  ckpt_save_interval=max(int(cfg['epochs'] / 10), 1),
                  lr_scheduler_each_iter=True,
                  starting_iteration=starting_iteration,
                  starting_epoch=starting_epoch)

    logger.info('Finished training')
    tb_logger.close()
    logger.info('**********************End**********************')