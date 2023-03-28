#!/usr/bin/env python
# coding=utf-8
# Author: Shuai Tao
# Mail: stao@create.aau.dk
# Create Time: Thu 14 Jul 2022 09:57:41 AM CEST

def get_len(train_file):
    
    count = -1
    for count, line in enumerate(open(train_file, 'rU')):
        pass
    count += 1
    print("File Length = %d" % count)
    
    return count


class hyperparameter():
    def __init__(self):
        self.save_path = '/user/create.aau.dk/zd77db/SPP/GRU_SPP/model_save'

        # Train
        self.train_input = '/user/create.aau.dk/zd77db/SPP/Encoder_test5/Data/train_input.txt'
        self.train_label = '/user/create.aau.dk/zd77db/SPP/Encoder_test5/Data/train_label.txt'
        self.train_file_len = get_len(self.train_input)

        self.total_mean = '/user/create.aau.dk/zd77db/SPP/Encoder_test5/Data/total_mean.pt'
        self.total_std = '/user/create.aau.dk/zd77db/SPP/Encoder_test5/Data/total_std.pt'

        # Validation
        self.validation_input = '/user/create.aau.dk/zd77db/SPP/Encoder_test5/Data/validation_input.txt'
        self.validation_label = '/user/create.aau.dk/zd77db/SPP/Encoder_test5/Data/validation_label.txt'
        self.val_file_len = get_len(self.validation_input)

        # Test
        self.test_file_path = '/user/create.aau.dk/zd77db/mini_dataset/test.csv'

        # Epoch
        self.epoch = 61

        #GRU
        self.input_size = 1
        self.hidden_size = 1
        self.num_layers = 1
        self.seq_len = 30
