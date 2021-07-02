# sys.path.append("./..")
# sys.path.append("/home/hu/Projects/planar_optical_flow")

import os
import argparse
from shutil import copyfile
import yaml
import torch

from src.utils.log_utils import create_logger, create_tb_logger
from src.depracted.model import BoundingBoxRegressor
# from src.data_handle.depracted.jrdb_dataset import JRDBDataset
from src.data_handle.jrdb_dataset import JRDBPointNet
from torch.utils.data import DataLoader
from src.utils.eval_utils import eval_Bb_regression, model_fn_eval_box_reg, eval_BB_reg_baseline


torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(description="arg parser")
parser.add_argument("--cfg", type=str, required=True, help="configuration of the experiment")
parser.add_argument("--ckpt", type=str, required=False, default=None)
args = parser.parse_args()

with open(args.cfg, 'r') as f:
    cfg = yaml.safe_load(f)
    cfg['name'] = os.path.basename(args.cfg).split(".")[0] + cfg['tag']
    print("Using configuration {}".format(cfg['name']))

if __name__ == "__main__":
    root_result_dir = os.path.join('../', 'output_pointnet', cfg['name'])
    os.makedirs(root_result_dir, exist_ok=True)
    # copyfile(args.cfg, os.path.join(root_result_dir, os.path.basename(args.cfg)))

    ckpt_dir = os.path.join(root_result_dir, 'ckpts')
    os.makedirs(ckpt_dir, exist_ok=True)

    logger= create_logger(root_result_dir)
    logger.info('**********************Start logging**********************')
    # log to file
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)

    logger.info('Prepare data')
    # test_set = JRDBDataset(data_path='/media/kevin/Kevin_Linux/data_backup/CV_data/JRDB_sample')
    test_set = JRDBPointNet(split='train', cfg=cfg)
    test_loader = DataLoader(test_set, batch_size=1, pin_memory=True,
                              num_workers=cfg['num_workers'], shuffle=False,
                              collate_fn=test_set.collate_batch)

    logger.info('Prepare model')
    model = BoundingBoxRegressor(input_dim=2, target_dim=3)
    model.cuda()


    #Load trained model
    cur_ckpt = "{}.pth".format(os.path.join(ckpt_dir, "ckpt_e{}".format(200)))
    # cur_ckpt = "{}.pth".format(os.path.join(ckpt_dir, "ckpt_e{}".format(200)))
    model.load_state_dict(torch.load(cur_ckpt)["model_state"])
    model.eval()

    #Testing
    logger.info("Start testing")
    print('Model evaluation')
    eval_Bb_regression(model, test_loader=test_loader, cfg=cfg, output_dir=root_result_dir)
    # eval_loss, loss_dim, loss_ori, iou = model_fn_eval_box_reg(model, test_loader)
    # logger.info('****************** Epoch Evaluation ******************')
    # logger.info('Validation, eval loss: {}'.format(eval_loss))
    # logger.info('Validation, dimension loss: {} [m]'.format(loss_dim))
    # logger.info('Validation, orientation loss: {} [rad]'.format(loss_ori))
    # logger.info('Validation, avg IOU: {}'.format(iou))
    # logger.info('********************* Epoch End **********************')
    # eval(model, test_loader=test_loader, cfg=cfg)

    # =======baseline=========
    # print('Baseline evaluation')
    # eval_BB_reg_baseline(test_set)

