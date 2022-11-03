#!/usr/bin/env python
# coding=utf-8
# Author: Shuai Tao
# Mail: stao@create.aau.dk
# Create Time: Thu 14 Jul 2022 11:11:00 AM CEST

import torch

from gru_module import GRU_REG
from hyperparameter import hyperparameter
from data_module import DNS_Loader
from train_module import  Model_Fit, get_no_params

if __name__ == "__main__":
    torch.manual_seed(0)


    para = hyperparameter()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    criterion = torch.nn.MSELoss()
    
    model_num = 1

    for n in range(model_num):
        print("-"*30, f"Model {n + 1}", "-"*30)

        model = GRU_REG(para, n).to(device)

        get_no_params(model)

        optimizer = torch.optim.Adam(params=model.parameters(),
                                       lr=1e-3,
                                       betas=(0.9, 0.999),
                                       eps=1e-8,
                                       weight_decay=0.00001,
                                       amsgrad=False)

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
        train_loader = DNS_Loader(para.train_input, para.train_label, para.train_file_len, n)
        val_loader = DNS_Loader(para.validation_input, para.validation_label, para.val_file_len, n)
        Model_Fit(model, criterion, optimizer, train_loader, val_loader, scheduler, device, para, n)

