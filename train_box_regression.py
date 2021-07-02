import argparse
import yaml
import torch
import wandb

from src.data_handle.get_dataloader import get_dataloader
from src.pipeline.pipeline import Pipeline
from src.model.get_model import get_model
import os

_DEBUG = False


def run_training(model, pipeline, cfg):
    # main train loop
    train_loader = get_dataloader(
        split="train", shuffle=True, dataset_cfg=cfg["dataset"], **cfg["dataloader"]
    )
    val_loader = get_dataloader(
        split="val", shuffle=True, dataset_cfg=cfg["dataset"], **cfg["dataloader"]
    )
    if _DEBUG:
        status = pipeline.train(model, train_loader, train_loader)
    else:
        status = pipeline.train(model, train_loader, val_loader)

    # test after training
    if not status:
        test_loader = get_dataloader(
            split="test",
            batch_size=1,
            num_workers=1,
            shuffle=False,
            dataset_cfg=cfg["dataset"],
        )
        pipeline.evaluate(model, test_loader, tb_prefix="TEST")

        # force wandb to push logs to server
        # https://github.com/wandb/client/issues/554
        wandb.log({"dummy": 1.0}, commit=False)


def run_evaluation(model, pipeline, cfg):
    val_loader = get_dataloader(
        split="val",
        batch_size=1,
        num_workers=1,
        shuffle=False,
        dataset_cfg=cfg["dataset"],
    )
    pipeline.evaluate(model, val_loader, tb_prefix="VAL")

    test_loader = get_dataloader(
        split="test",
        batch_size=1,
        num_workers=1,
        shuffle=False,
        dataset_cfg=cfg["dataset"],
    )
    pipeline.evaluate(model, test_loader, tb_prefix="TEST")

    # force wandb to push logs to server
    # https://github.com/wandb/client/issues/554
    wandb.log({"dummy": 1.0}, commit=False)


if __name__ == "__main__":
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

    root_result_dir = cfg["pipeline"]["Logger"]["result_dir"]
    os.makedirs(root_result_dir, exist_ok=True)

    project = "box_reg" if not args.tmp else "tmp"
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

    # training or evaluation
    if not args.evaluation:
        run_training(model, pipeline, cfg)
    else:
        run_evaluation(model, pipeline, cfg)

    pipeline.close()
