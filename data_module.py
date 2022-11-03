#!/usr/bin/env python
# coding=utf-8
# Author: Shuai Tao
# Mail: stao@create.aau.dk
# Create Time: Thu 14 Jul 2022 10:03:33 AM CEST

import numpy as np
from torch.utils.data import Dataset, DataLoader


class DNS_Dataset(Dataset):
    def __init__(self, input_files, label_files, file_len, num):
        self.input_data = np.loadtxt(input_files)
        self.label_data = np.loadtxt(label_files)
        self.len = int(file_len / 129)
        self.num = num


    def __getitem__(self, index):
        
        
        #Frequency bin-wise model

        #if self.num < 1:
        #    return self.input_data[index * 129 + self.num : index * 129 + self.num + 2, :], \
        #           self.label_data[index * 129 + self.num, :]
        #elif self.num > 127:
        #    return self.input_data[index * 129 + self.num - 1 : index * 129 + self.num + 1, :], \
        #           self.label_data[index * 129 + self.num, :]
        #elif (self.num > 0 and self.num < 2) or (self.num < 128 and self.num > 126):
        #    return self.input_data[index * 129 + self.num - 1 : index * 129 + self.num + 2, :], \
        #           self.label_data[index * 129 + self.num, :]
        #elif (self.num == 2) or (self.num == 126):
        #    return self.input_data[index * 129 + self.num - 2 : index * 129 + self.num + 3, :], \
        #           self.label_data[index * 129 + self.num, :]
        #elif (self.num == 3) or (self.num == 125):
        #    return self.input_data[index * 129 + self.num - 3 : index * 129 + self.num + 4, :], \
        #           self.label_data[index * 129 + self.num, :]
        #else:
        #    return self.input_data[index * 129 + self.num - 4 : index * 129 + self.num + 5, :], \
        #           self.label_data[index * 129 + self.num, :]
 
        
        
        #Typical DNN-based model
        return self.input_data[index * 129 : (index + 1) * 129, :], \
               self.label_data[index * 129 : (index + 1) * 129, :]


        


    def __len__(self):
        return self.len


def DNS_Loader(input_file, label_file, file_len, num):

    dataset = DNS_Dataset(input_file, label_file, file_len, num)

    loader = DataLoader(dataset = dataset,
                        batch_size = 128,
                        shuffle = True,
                        num_workers = 4,
                        )
    return loader
