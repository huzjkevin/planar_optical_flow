import sys
# sys.path.append("/home/kevin/HIWI/CV/planar_optical_flow")
# sys.path.append("/home/hu/Projects/planar_optical_flow")
# print(sys.path)

import os
import argparse
from shutil import copyfile
import yaml
import torch
import torchvision
from torch import optim

from src.utils.train_utils import Trainer, load_checkpoint, LucasScheduler
from src.utils.eval_utils import model_fn_dr_spaam, model_fn_eval
import src.utils.train_utils as tu
from src.utils.dataset_dr_spaam import create_dataloader
from src.depracted.model import SpatialDROW, FlowDROW_pretrained
import src.utils.utils as u
from src.utils.log_utils import create_logger, create_tb_logger


torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(description="arg parser")
parser.add_argument("--cfg", type=str, required=True, help="configuration of the experiment")
parser.add_argument("--ckpt", type=str, required=False, default=None)
args = parser.parse_args()

with open(args.cfg, 'r') as f:
    cfg = yaml.safe_load(f)
    cfg['name'] = os.path.basename(args.cfg).split(".")[0] + cfg['tag']

if __name__ == '__main__':
    root_result_dir = os.path.join('../', 'output_test', cfg['name'])
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

    train_loader, eval_loader = create_dataloader(data_path="../data/DROWv2-data",
                                                  num_scans=cfg['num_scans'],
                                                  batch_size=cfg['batch_size'],
                                                  num_workers=cfg['num_workers'],
                                                  network_type=cfg['network'],
                                                  train_with_val=cfg['train_with_val'],
                                                  use_data_augumentation=cfg['use_data_augumentation'],
                                                  cutout_kwargs=cfg['cutout_kwargs'],
                                                  polar_grid_kwargs=cfg['polar_grid_kwargs'],
                                                  pedestrian_only=cfg['pedestrian_only'])

    logger.info('Prepare model')
    model = FlowDROW_pretrained(num_scans=cfg['num_scans'],
                                num_pts=cfg['cutout_kwargs']['num_cutout_pts'],
                                focal_loss_gamma=cfg['focal_loss_gamma'],
                                alpha=cfg['similarity_kwargs']['alpha'],
                                window_size=cfg['similarity_kwargs']['window_size'],
                                pedestrian_only=cfg['pedestrian_only'])

    model.cuda()

    #Prepare training
    logger.info('Prepare training')
    # optimizer = optim.Adam(model.parameters(), lr=tu.lr_scheduler())
    optimizer = optim.Adam(model.parameters(), amsgrad=True)

    if 'lr_kwargs' in cfg:
        e0, e1 = cfg['lr_kwargs']['e0'], cfg['lr_kwargs']['e1']
    else:
        e0, e1 = 0, cfg['epochs']

    lr_scheduler = LucasScheduler(optimizer, 0, 1e-2, cfg['epochs'], 1e-5)

    #Logging
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
                      model_fn_dr_spaam,
                      optimizer,
                      model_fn_eval=model_fn_eval,
                      ckpt_dir=ckpt_dir,
                      lr_scheduler=lr_scheduler,
                      grad_norm_clip=cfg['grad_norm_clip'],
                      tb_logger=tb_logger,
                      logger=logger)

    trainer.train(num_epochs=cfg['epochs'],
                  train_loader=train_loader,
                  eval_loader=eval_loader,
                  eval_frequency=max(int(cfg['epochs'] / 20), 1),
                  ckpt_save_interval=max(int(cfg['epochs'] / 10), 1),
                  lr_scheduler_each_iter=True,
                  starting_iteration=starting_iteration,
                  starting_epoch=starting_epoch)

    logger.info('Finished training')
    tb_logger.close()
    logger.info('**********************End**********************')