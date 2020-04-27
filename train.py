# -*- coding: utf-8 -*-
# file: train.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2018. All Rights Reserved.
import torch

from helpers.argparser import ArgParser
from helpers.seeder import Seeder
from model_runner import ModelRunner
from yaml import load, FullLoader
from zoo import initializers, models, optimizers

with open('config.yml', 'r') as f:
    config = load(f, Loader=FullLoader)


def main():
    # get hyperparameters from input args
    opt = ArgParser().get_options()

    # deterministic seed across numpy, torch and cuda
    # store as variable due to garbage collecting
    seeder = Seeder(opt.seed)
    seeder.activate()

    # data from ./zoo.py
    opt.model_class = models[opt.model_name]
    opt.initializer = initializers[opt.initializer]
    opt.optimizer = optimizers[opt.optimizer]
    # data from ./config.yml
    opt.dataset_file = config['datasets'][opt.dataset]
    opt.model_inputs = config['model_inputs'][opt.model_name]

    # run on gpu if available
    device = opt.device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    opt.device = torch.device(device)

    runner = ModelRunner(opt)
    runner.run()


if __name__ == '__main__':
    main()
