import sys
sys.path.append("./..")
# print(sys.path)
import torch
import torchvision
import os
from torch import optim
from src.utils.train_utils import Trainer, load_checkpoint
from src.utils.eval_utils import model_fn, model_fn_eval, eval
import src.utils.train_utils as tu
from src.utils.dataset import FlowDataset, FlowDatasetTmp, FlowDatasetTmp2, FlowDataset2
from src.utils.dataset_dr_spaam import create_dataloader, DROWDataset
from src.depracted.model import Prototype, PrototypeTest
# from src.utils.log_utils import create_tb_logger
import src.utils.utils as u
from src.utils.train_utils import create_tb_logger
from torch.utils.data import SequentialSampler

if __name__ == "__main__":
    #TODO: Simplifiy and automate the process
    #Create directory for storing results
    output_dir = os.path.join("./..", "output")
    os.makedirs(output_dir, exist_ok=True)
    ckpt_dir = os.path.join(output_dir, "ckpts")
    os.makedirs(ckpt_dir, exist_ok=True)

    #some cfgs, some cfg will be used in the future
    #TODO::put all kinds of cfgs and hyperparameter into a config file. e.g. yaml
    cfg = {}
    cfg["ckpt"] = None
    cfg["num_epochs"] = 100
    cfg["batch_size"] = 4
    cfg["pdrop"] = 0.1
    cfg["num_workers"] = 4

    print("Prepare data")
    # test_dataset = FlowDataset(data_path='./../data/DROWv2-data')
    # test_dataset = FlowDatasetTmp(data_path='./../data/DROWv2-data', split="train", testing=True)
    # test_dataset = FlowDatasetTmp2(data_path='./../data/DROWv2-data', split="test", testing=True)
    test_dataset = FlowDataset2(data_path='./../data/DROWv2-data', split="test", testing=True)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                                batch_size=cfg["batch_size"],
                                                num_workers=cfg["num_workers"],
                                                shuffle=False,
                                                collate_fn=test_dataset.collate_batch)
                                                
    # sample = iter(train_dataset)
    # rtn_dict = next(sample)

    #Prepare model
    print("Prepare model")

    model = Prototype(in_channel=2)
    # model = PrototypeTest(in_channel=2)
    # model = Prototype(in_channel=2, max_displacement=1)
    model.cuda()

    #Load trained model
    cur_ckpt = "{}.pth".format(os.path.join(ckpt_dir, "ckpt_e{}".format(cfg["num_epochs"])))
    model.load_state_dict(torch.load(cur_ckpt)["model_state"])
    model.eval()

    #Testing
    print("Start testing")
    eval(model, test_loader=test_loader, cfg=cfg, output_dir=output_dir)
    # eval(model, test_loader=test_loader, cfg=cfg)
