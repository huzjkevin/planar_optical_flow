import os
import shutil
import time
import yaml

_OUTPUT_DIR = "./experiments"


"""
Convinent functions
"""


def _get_default_sbatch_args_claix(job_name, log_tag):
    return {
        "job-name": job_name,
        "output": f"/home/kq708907/slurm_logs/%x_%J_{log_tag}.log",
        "mail-type": "ALL",
        "mail-user": "huzjkevin@gmail.com",
        "cpus-per-task": "8",
        "mem-per-cpu": "3G",
        "gres": "gpu:1",
        "time": "2-00:00:00",
        "signal": "TERM@120",
        "partition": "c18g",
    }


def _get_default_sbatch_args_vision(job_name, log_tag, partition="lopri"):
    """
    Args:
        partition (str, optional): "lopri", "hipri", "trtx-lo", or "trtx-hi".
            Defaults to "lopri".
    """
    return {
        "job-name": job_name,
        "output": f"/home/hu/Projects/slurm_logs/%x_%J_{log_tag}.log",
        "mail-type": "ALL",
        "mail-user": "huzjkevin@gmail.com",
        "partition": partition,
        "cpus-per-task": "4",
        "mem": "16G",
        "gres": "gpu:1",
        "time": "2-00:00:00",
        "signal": "TERM@120",
    }


def _write_sbatch_file_vision(fname, sbatch_args, cfg_fname, cmd_args=""):
    """
    Args:
        fname (str): Output file, full path
        sbatch_args (dict):
        cfg_fname (str): cfg file for training
        cmd_args (str, optional): Additional cmd arguments. Defaults to "".
    """
    cfg_fname = os.path.abspath(cfg_fname)

    lines = [
        "cd $HOME/Projects/planar_optical_flow\n",
        "wandb on\n",
        f"srun --unbuffered python ./train_box_regression.py --cfg {cfg_fname} {cmd_args}\n",
    ]

    with open(fname, "w") as f:
        f.write("#!/bin/bash\n")

        for key, val in sbatch_args.items():
            f.write(f"#SBATCH --{key}={val}\n")

        for li in lines:
            f.write(li)


def _write_sbatch_file_claix(fname, sbatch_args, cfg_fname, cmd_args=""):
    cfg_fname = os.path.abspath(cfg_fname)

    lines = [
        "source $HOME/.zshrc\n",
        "conda activate kevin\n",
        "cd $HOME/Projects/planar_optical_flow\n",
        "wandb on\n",
        f"srun --unbuffered python ./train_box_regression.py --cfg {cfg_fname} {cmd_args}\n",
    ]

    with open(fname, "w") as f:
        f.write("#!/usr/local_rwth/bin/zsh\n")

        for key, val in sbatch_args.items():
            f.write(f"#SBATCH --{key}={val}\n")

        for li in lines:
            f.write(li)


def _write_experiment_files(exp_dir, cfgs, names, names_short, cmd_args=None):
    if cmd_args is None:
        cmd_args = [""] * len(cfgs)

    for c, n, n_abb, ca in zip(cfgs, names, names_short, cmd_args):
        c["pipeline"]["Logger"]["tag"] = n

        yaml_file = os.path.join(exp_dir, f"{n}.yaml")
        with open(yaml_file, "w") as f:
            yaml.dump(c, f)

        sbatch_args_claix = _get_default_sbatch_args_claix(n_abb, n)
        sh_file_claix = os.path.join(exp_dir, f"{n}_claix.sh")
        _write_sbatch_file_claix(sh_file_claix, sbatch_args_claix, yaml_file, ca)

        sbatch_args_vision = _get_default_sbatch_args_vision(n_abb, n)
        sh_file_vision = os.path.join(exp_dir, f"{n}_vision.sh")
        _write_sbatch_file_vision(sh_file_vision, sbatch_args_vision, yaml_file, ca)

    # avoid some weird errors relating to remove and create files (directory not empty)
    time.sleep(0.1)


def _mkdir_reset(target_dir):
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)

    os.makedirs(target_dir, exist_ok=False)


"""
Experiments
"""


def train_3d_box_regression_different_epochs():
    exp_name = "3d_box_reg_different_epochs"
    exp_dir = os.path.join(_OUTPUT_DIR, exp_name)
    _mkdir_reset(exp_dir)

    with open("./config/train_3d_box_regression.yaml", "r") as f:
        cfg1 = yaml.safe_load(f)
    with open("./config/train_3d_box_regression.yaml", "r") as f:
        cfg2 = yaml.safe_load(f)
    with open("./config/train_3d_box_regression.yaml", "r") as f:
        cfg3 = yaml.safe_load(f)

    cfg1["pipeline"]["Trainer"]["epoch"] = 20
    cfg1["pipeline"]["Optim"]["scheduler_kwargs"]["epoch1"] = 20

    cfg2["pipeline"]["Trainer"]["epoch"] = 40
    cfg2["pipeline"]["Optim"]["scheduler_kwargs"]["epoch1"] = 40

    cfg3["pipeline"]["Trainer"]["epoch"] = 100
    cfg3["pipeline"]["Optim"]["scheduler_kwargs"]["epoch1"] = 100

    cfgs = [cfg1, cfg2, cfg3]
    names = ["3d_box_regression_e20", "3d_box_regression_e40", "3d_box_regression_e100"]
    names_short = ["3d_box_e20", "3d_box_e40", "3d_box_e100"]
    # cmd_args = ["--cfg /media/kevin/work/HIWI/CV/planar_optical_flow/config/train_3d_box_regression.yaml"]

    _write_experiment_files(exp_dir, cfgs, names, names_short)


def train_3d_box_regression_different_batch_size():
    exp_name = "3d_box_reg_different_batch_size"
    exp_dir = os.path.join(_OUTPUT_DIR, exp_name)
    _mkdir_reset(exp_dir)

    with open("./config/train_3d_box_regression.yaml", "r") as f:
        cfg1 = yaml.safe_load(f)
    with open("./config/train_3d_box_regression.yaml", "r") as f:
        cfg2 = yaml.safe_load(f)
    with open("./config/train_3d_box_regression.yaml", "r") as f:
        cfg3 = yaml.safe_load(f)

    cfg1["dataloader"]["batch_size"] = 64

    cfg2["dataloader"]["batch_size"] = 128

    cfg3["dataloader"]["batch_size"] = 256

    cfgs = [cfg1, cfg2, cfg3]
    names = [
        "3d_box_regression_bs64",
        "3d_box_regression_bs128",
        "3d_box_regression_bs256",
    ]
    names_short = ["3d_box_bs64", "3d_box_bs128", "3d_box_bs256"]
    # cmd_args = ["--cfg /media/kevin/work/HIWI/CV/planar_optical_flow/config/train_3d_box_regression.yaml"]

    _write_experiment_files(exp_dir, cfgs, names, names_short)


def train_3d_box_regression_data_augmentation():
    exp_name = "3d_box_reg_data_augmentation"
    exp_dir = os.path.join(_OUTPUT_DIR, exp_name)
    _mkdir_reset(exp_dir)

    with open("./config/train_3d_box_regression.yaml", "r") as f:
        cfg1 = yaml.safe_load(f)
    with open("./config/train_3d_box_regression.yaml", "r") as f:
        cfg2 = yaml.safe_load(f)

    cfg1["dataset"]["augmentation_kwargs"]["use_data_augmentation"] = True

    cfg2["dataset"]["augmentation_kwargs"]["use_data_augmentation"] = False

    cfgs = [cfg1, cfg2]
    names = [
        "3d_box_regression_use_data_augmentation",
        "3d_box_regression_no_data_augmentation",
    ]
    names_short = ["3d_box_data_aug_yes", "3d_box_data_aug_no"]

    # cmd_args = ["--cfg /media/kevin/work/HIWI/CV/planar_optical_flow/config/train_3d_box_regression.yaml"]

    _write_experiment_files(exp_dir, cfgs, names, names_short)


def train_3d_box_regression_different_input_size():
    exp_name = "3d_box_reg_different_input_size"
    exp_dir = os.path.join(_OUTPUT_DIR, exp_name)
    _mkdir_reset(exp_dir)

    with open("./config/train_3d_box_regression.yaml", "r") as f:
        cfg1 = yaml.safe_load(f)
    with open("./config/train_3d_box_regression.yaml", "r") as f:
        cfg2 = yaml.safe_load(f)
    with open("./config/train_3d_box_regression.yaml", "r") as f:
        cfg3 = yaml.safe_load(f)
    with open("./config/train_3d_box_regression.yaml", "r") as f:
        cfg4 = yaml.safe_load(f)

    cfg1["dataset"]["input_size"] = 64
    cfg2["dataset"]["input_size"] = 128
    cfg3["dataset"]["input_size"] = 256
    cfg4["dataset"]["input_size"] = 512

    cfgs = [cfg1, cfg2, cfg3, cfg4]
    names = [
        "3d_box_regression_input64",
        "3d_box_regression_input128",
        "3d_box_regression_input256",
        "3d_box_regression_input512",
    ]
    names_short = [
        "3d_box_input64",
        "3d_box_input128",
        "3d_box_input256",
        "3d_box_input512",
    ]

    # cmd_args = ["--cfg /media/kevin/work/HIWI/CV/planar_optical_flow/config/train_3d_box_regression.yaml"]

    _write_experiment_files(exp_dir, cfgs, names, names_short)


def train_3d_box_regression_different_min_segment():
    exp_name = "3d_box_reg_different_min_segment"
    exp_dir = os.path.join(_OUTPUT_DIR, exp_name)
    _mkdir_reset(exp_dir)

    with open("./config/train_3d_box_regression.yaml", "r") as f:
        cfg1 = yaml.safe_load(f)
    with open("./config/train_3d_box_regression.yaml", "r") as f:
        cfg2 = yaml.safe_load(f)
    with open("./config/train_3d_box_regression.yaml", "r") as f:
        cfg3 = yaml.safe_load(f)
    with open("./config/train_3d_box_regression.yaml", "r") as f:
        cfg4 = yaml.safe_load(f)

    cfg1["dataset"]["min_segment_size"] = 5
    cfg2["dataset"]["min_segment_size"] = 10
    cfg3["dataset"]["min_segment_size"] = 20
    cfg4["dataset"]["min_segment_size"] = 50

    cfgs = [cfg1, cfg2, cfg3, cfg4]
    names = [
        "3d_box_regression_min_segment5",
        "3d_box_regression_min_segment10",
        "3d_box_regression_min_segment20",
        "3d_box_regression_min_segment50",
    ]
    names_short = [
        "3d_box_minseg5",
        "3d_box_minseg10",
        "3d_box_minseg20",
        "3d_box_minseg50",
    ]

    # cmd_args = ["--cfg /media/kevin/work/HIWI/CV/planar_optical_flow/config/train_3d_box_regression.yaml"]

    _write_experiment_files(exp_dir, cfgs, names, names_short)


def train_3d_box_regression_different_dropout():
    exp_name = "3d_box_reg_different_dropout"
    exp_dir = os.path.join(_OUTPUT_DIR, exp_name)
    _mkdir_reset(exp_dir)

    with open("./config/train_3d_box_regression.yaml", "r") as f:
        cfg1 = yaml.safe_load(f)
    with open("./config/train_3d_box_regression.yaml", "r") as f:
        cfg2 = yaml.safe_load(f)
    with open("./config/train_3d_box_regression.yaml", "r") as f:
        cfg3 = yaml.safe_load(f)

    cfg1["model"]["dropout"] = 0.1
    cfg2["model"]["dropout"] = 0.3
    cfg3["model"]["dropout"] = 0.5

    cfgs = [cfg1, cfg2, cfg3]
    names = [
        "3d_box_regression_dropout01",
        "3d_box_regression_dropout03",
        "3d_box_regression_dropout05",
    ]
    names_short = ["3d_box_dp01", "3d_box_dp03", "3d_box_dp05"]

    # cmd_args = ["--cfg /media/kevin/work/HIWI/CV/planar_optical_flow/config/train_3d_box_regression.yaml"]

    _write_experiment_files(exp_dir, cfgs, names, names_short)


def train_3d_box_regression_different_segment_radius():
    exp_name = "3d_box_reg_different_segment_radius"
    exp_dir = os.path.join(_OUTPUT_DIR, exp_name)
    _mkdir_reset(exp_dir)

    with open("./config/train_3d_box_regression.yaml", "r") as f:
        cfg1 = yaml.safe_load(f)
    with open("./config/train_3d_box_regression.yaml", "r") as f:
        cfg2 = yaml.safe_load(f)
    with open("./config/train_3d_box_regression.yaml", "r") as f:
        cfg3 = yaml.safe_load(f)
    with open("./config/train_3d_box_regression.yaml", "r") as f:
        cfg4 = yaml.safe_load(f)

    cfg1["dataset"]["radius_segment"] = 0.25
    cfg2["dataset"]["radius_segment"] = 0.4
    cfg3["dataset"]["radius_segment"] = 0.7
    cfg4["dataset"]["radius_segment"] = 1.0

    cfgs = [cfg1, cfg2, cfg3, cfg4]
    names = [
        "3d_box_regression_segment_radius025",
        "3d_box_regression_segment_radius04",
        "3d_box_regression_segment_radius07",
        "3d_box_regression_segment_radius10",
    ]
    names_short = ["3d_box_segR025", "3d_box_segR04", "3d_box_segR07", "3d_box_segR10"]

    # cmd_args = ["--cfg /media/kevin/work/HIWI/CV/planar_optical_flow/config/train_3d_box_regression.yaml"]

    _write_experiment_files(exp_dir, cfgs, names, names_short)


if __name__ == "__main__":
    train_3d_box_regression_different_epochs()
    train_3d_box_regression_different_batch_size()
    train_3d_box_regression_data_augmentation()
    train_3d_box_regression_different_input_size()
    train_3d_box_regression_different_min_segment()
    train_3d_box_regression_different_segment_radius()
    train_3d_box_regression_different_dropout()
