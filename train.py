# -*- coding: utf-8 -*-
# file: train.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2018. All Rights Reserved.
import random
from argparser import ArgParser

import numpy
import torch
from yaml import load
from zoo import models, optimizers, initializers
from model_runner import ModelRunner

with open('config.yml', 'r') as f:
    config = load(f)


def main():
    opt = ArgParser().get_options()
    # Hyper Parameters
    if opt.seed is not None:
        random.seed(opt.seed)
        numpy.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed(opt.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    opt.model_class = models[opt.model_name]
    opt.dataset_file = config['datasets'][opt.dataset]
    opt.model_inputs = config['model_inputs'][opt.model_name]
    opt.initializer = initializers[opt.initializer]
    opt.optimizer = optimizers[opt.optimizer]

    device = opt.device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    opt.device = torch.device(device)

    runner = ModelRunner(opt)
    runner.run()


if __name__ == '__main__':
    main()
