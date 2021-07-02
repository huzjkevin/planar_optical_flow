import sys
# sys.path.append("./..")
# sys.path.append("/home/hu/Projects/planar_optical_flow")

import torch
import torchvision
import os
import argparse
import yaml
from shutil import copyfile

from torch import optim
from src.utils.train_utils import Trainer, load_checkpoint
from src.utils.eval_utils import model_fn, model_fn_eval, eval_dr_spaam
import src.utils.train_utils as tu
from src.utils.dataset import FlowDataset, FlowDatasetTmp, FlowDatasetTmp2, FlowDataset2
from src.utils.dataset_dr_spaam import create_test_dataloader, create_dataloader, DROWDataset2
from src.depracted.model import SpatialDROW, FlowDROW_pretrained
from torch.utils.data import DataLoader

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(description="arg parser")
parser.add_argument("--cfg", type=str, required=True, help="configuration of the experiment")
parser.add_argument("--ckpt", type=str, required=False, default=None)
args = parser.parse_args()

with open(args.cfg, 'r') as f:
    cfg = yaml.safe_load(f)
    cfg['name'] = os.path.basename(args.cfg).split(".")[0] + cfg['tag']

if __name__ == "__main__":
    #Create directory for storing results
    root_result_dir = os.path.join('../', 'output_test', cfg['name'])
    os.makedirs(root_result_dir, exist_ok=True)
    copyfile(args.cfg, os.path.join(root_result_dir, os.path.basename(args.cfg)))

    ckpt_dir = os.path.join(root_result_dir, 'ckpts')
    os.makedirs(ckpt_dir, exist_ok=True)

    print("Prepare data")

    test_loader = create_test_dataloader(data_path="../data/DROWv2-data",
                                         num_scans=cfg['num_scans'],
                                         network_type=cfg['network'],
                                         cutout_kwargs=cfg['cutout_kwargs'],
                                         polar_grid_kwargs=cfg['polar_grid_kwargs'],
                                         pedestrian_only=cfg['pedestrian_only'],
                                         split='train')

    #Prepare model
    print("Prepare model")

    model = FlowDROW_pretrained(num_scans=cfg['num_scans'],
                                num_pts=cfg['cutout_kwargs']['num_cutout_pts'],
                                focal_loss_gamma=cfg['focal_loss_gamma'],
                                alpha=cfg['similarity_kwargs']['alpha'],
                                window_size=cfg['similarity_kwargs']['window_size'],
                                pedestrian_only=cfg['pedestrian_only'])

    # pre_trained_ckpt = './pre_trained_ckpts/dr_spaam_e40.pth'
    # load_checkpoint(model=model.dr_spaam, filename=pre_trained_ckpt) # Question: should I also load this during evaluation? Or I just load the trained ckpt?

    model.cuda()

    #Load trained model
    cur_ckpt = "{}.pth".format(os.path.join(ckpt_dir, "ckpt_e{}".format(cfg["epochs"])))
    # cur_ckpt = "{}.pth".format(os.path.join(ckpt_dir, "ckpt_e{}".format(200)))
    model.load_state_dict(torch.load(cur_ckpt)["model_state"])
    model.eval()

    #Testing
    print("Start testing")
    eval_dr_spaam(model, test_loader=test_loader, cfg=cfg, output_dir=root_result_dir)
    # eval(model, test_loader=test_loader, cfg=cfg)
