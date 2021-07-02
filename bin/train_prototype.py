import sys
sys.path.append("/home/kevin/HIWI/CV/planar_optical_flow")
# print(sys.path)
import torch
import torchvision
import os
print(os.getcwd())
from torch import optim
from src.utils.train_utils import Trainer, load_checkpoint
from src.utils.eval_utils import model_fn, model_fn_eval
import src.utils.train_utils as tu
from src.utils.dataset import FlowDataset, FlowDatasetTmp, FlowDatasetTmp2, FlowDataset2
from src.depracted.model import Prototype, PrototypeTest
# from src.utils.log_utils import create_tb_logger
import src.utils.utils as u
from src.utils.train_utils import create_tb_logger


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
    cfg["num_epochs"] = 300
    cfg["batch_size"] = 100
    cfg["pdrop"] = 0.1
    cfg["ckpt_save_interval"] = 100
    cfg["grad_norm_clip"] = None
    cfg["num_workers"] = 4

    print("Prepare data")
    # train_dataset = FlowDataset(data_path='./../data/DROWv2-data')
    # train_dataset = FlowDatasetTmp(data_path='./../data/DROWv2-data')
    train_dataset = FlowDatasetTmp2(data_path='./../data/DROWv2-data')
    # train_dataset = FlowDataset2(data_path='./../data/DROWv2-data')
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=cfg["batch_size"],
                                               num_workers=cfg["num_workers"],
                                               shuffle=True,
                                               collate_fn=train_dataset.collate_batch)

    # sample = iter(train_dataset)
    # rtn_dict = next(sample)

    #Prepare model
    print("Prepare model")

    model = Prototype(in_channel=2)
    # model = PrototypeTest(in_channel=2)
    # model = Prototype(in_channel=2, max_displacement=1)
    model.cuda()

    #Prepare training
    print("Prepare training")
    optimizer = optim.Adam(model.parameters(), lr=tu.lr_scheduler())

    #Define starting iteration/epochs.
    #Will use checkpoints in the future when running on clusters
    if cfg["ckpt"] is not None:
        starting_iteration, starting_epoch = load_checkpoint(model=model, optimizer=optimizer, filename=cfg["ckpt"])
    elif os.path.isfile(os.path.join(ckpt_dir, "sigterm_ckpt.pth")):
        starting_iteration, starting_epoch = load_checkpoint(model=model, optimizer=optimizer, filename=os.path.join(ckpt_dir, "sigterm_ckpt.pth"))
    else:
        starting_iteration, starting_epoch = 0, 0

    #Logging
    tb_logger = create_tb_logger(output_dir)

    #Training
    print("Start training")

    trainer = Trainer(model=model,
                          model_fn=model_fn,
                          model_fn_eval=model_fn_eval,
                          optimizer=optimizer,
                          ckpt_dir=ckpt_dir,
                          grad_norm_clip=cfg["grad_norm_clip"],
                          tb_logger=tb_logger)

    trainer.train(num_epochs=cfg["num_epochs"],
                      train_loader=train_loader,
                      ckpt_save_interval=cfg["ckpt_save_interval"],
                      starting_iteration=starting_iteration,
                      starting_epoch=starting_epoch)


    #Finalizing
    print("Analysis finished\n")
    #TODO: integrate logging, visualiztion, GPU data parallel etc in the future