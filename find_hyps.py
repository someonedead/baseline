import os
import sys
import yaml
import random
import argparse
import os.path as osp

import torch
import numpy as np
from tqdm import tqdm

import utils
from models import models
from data import get_dataloader
from train import train, validation

# from utils import convert_dict_to_tuple
from collections import namedtuple

import logging
import sys

import optuna


def convert_dict_to_tuple(dictionary):
    for key, value in dictionary.items():
        if isinstance(value, dict):
            dictionary[key] = convert_dict_to_tuple(value)
    return namedtuple("GenericDict", dictionary.keys())(**dictionary)


def objective(trial) -> None:
    """
    Run train process of classification model
    :param args: all parameters necessary for launch
    :return: None
    """
    args = parse_arguments(sys.argv[1:])
    with open(args.cfg) as f:
        data = yaml.safe_load(f)

    # data["model"]["arch"] = "efficientnet_lite0"
    data["train"]["arcface"]["s"] = trial.suggest_int("s", 8, 15, step=1)
    data["train"]["arcface"]["m"] = trial.suggest_float("m", 0.4, 0.52, step=0.01)
    data["train"]["learning_rate"] = trial.suggest_float("lr", 1e-5, 5e-4)
    data["train"]["weight_decay"] = trial.suggest_float("weight_decay", 1e-5, 5e-3)
    data["model"]["dropout"] = trial.suggest_float("dropout", 0, 0.1, step=0.01)
    # data["dataset"]["padding"] = trial.suggest_int("padding", 0, 40, step=1)

    # padding = data["dataset"]["padding"]
    weight_decay = data["train"]["weight_decay"]
    dropout = data["model"]["dropout"]
    s = data["train"]["arcface"]["s"]
    m = data["train"]["arcface"]["m"]
    lr = data["train"]["learning_rate"]
    print(
        f"Start with s: {s} m: {m} lr: {lr:.6f} weight_decay: {weight_decay} dropout: {dropout}"  # padding: {padding}"
    )
    config = convert_dict_to_tuple(data)

    seed = config.dataset.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    outdir = osp.join(config.outdir, config.exp_name)

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    train_loader, val_loader = get_dataloader.get_dataloaders(config)

    net = models.load_model(config)
    if config.num_gpu > 1:
        net = torch.nn.DataParallel(net)

    criterion, optimizer, scheduler = utils.get_training_parameters(config, net)
    train_epoch = tqdm(
        range(config.train.n_epoch), dynamic_ncols=True, desc="Epochs", position=0
    )

    # main process
    best_acc = 0.0
    # for epoch in train_epoch:
    train(net, train_loader, criterion, optimizer, config, 0)
    acc = validation(
        net,
        val_loader=val_loader,
        criterion=criterion,
        epoch=0,
    )
    return acc


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="config/base.yml", help="Path to config file.")
    return parser.parse_args(argv)


def print_best_callback(study, trial):
    print(f"Best value: {study.best_value}, Best params: {study.best_trial.params}")


def main():
    # Add stream handler of stdout to show the messages
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study_name = "find_s_m_lr_wd_do"  # Unique identifier of the study.
    storage_name = "sqlite:///{}.db".format(study_name)
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        load_if_exists=True,
        direction="maximize",
    )
    study.optimize(objective, n_trials=1000, callbacks=[print_best_callback])


if __name__ == "__main__":
    main()
