import argparse
import yaml
import torch
import wandb

from dr_spaam.dr_spaam.data_handle.drow_dataset import get_dataloader
from dr_spaam.dr_spaam.pipeline.pipeline import Pipeline
from dr_spaam.dr_spaam.model.get_model import get_model


# Run benchmark to select fastest implementation of ops.
torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(description="arg parser")
parser.add_argument(
    "--cfg", type=str, required=True, help="configuration of the experiment"
)
parser.add_argument("--ckpt", type=str, required=False, default=None)
parser.add_argument("--cont", default=False, action="store_true")
parser.add_argument("--tmp", default=False, action="store_true")
parser.add_argument("--evaluation", default=False, action="store_true")
args = parser.parse_args()

with open(args.cfg, "r") as f:
    cfg = yaml.safe_load(f)
    cfg["pipeline"]["Logger"]["backup_list"].append(args.cfg)
    if args.tmp:
        cfg["pipeline"]["Logger"]["tag"] += "_TMP"

project = "dr_spaam" if not args.tmp else "tmp"
wandb.init(
    project=project, name=cfg["pipeline"]["Logger"]["tag"], sync_tensorboard=True
)
wandb.config.update(cfg)

model = get_model(cfg["model"])
model.cuda()
wandb.watch(model, log="all")

pipeline = Pipeline(model, cfg["pipeline"])

if args.ckpt:
    pipeline.load_ckpt(model, args.ckpt)
elif args.cont and pipeline.sigterm_ckpt_exists():
    pipeline.load_sigterm_ckpt(model)

# training
if not args.evaluation:
    # main train loop
    train_loader = get_dataloader(
        split="train", cfg=cfg["dataset"], **cfg["dataloader"]
    )
    val_loader = get_dataloader(split="val", cfg=cfg["dataset"], **cfg["dataloader"])
    status = pipeline.train(model, train_loader, val_loader)

    # test after training
    if not status:
        test_loader = get_dataloader(
            split="test", batch_size=1, num_workers=1, cfg=cfg["dataset"]
        )
        pipeline.evaluate(model, test_loader, tb_prefix="TEST")

# evaluation
else:
    val_loader = get_dataloader(
        split="val", batch_size=1, num_workers=1, cfg=cfg["dataset"]
    )
    pipeline.evaluate(model, val_loader, tb_prefix="VAL")

    test_loader = get_dataloader(
        split="test", batch_size=1, num_workers=1, cfg=cfg["dataset"]
    )
    pipeline.evaluate(model, test_loader, tb_prefix="TEST")

pipeline.close()

# force wandb to push logs to server
# https://github.com/wandb/client/issues/554
wandb.log({"dummy": 1.0}, commit=False)
