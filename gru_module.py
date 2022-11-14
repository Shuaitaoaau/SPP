#!/usr/bin/env python
# coding=utf-8
# Author: Shuai Tao
# Mail: stao@create.aau.dk
# Create Time: Thu 11 Aug 2022 11:09:14 AM CEST



import torch.nn as nn

class GRU_REG(nn.Module):
    def __init__(self, para, n):
        
        #Typical DNN-based model
        self.input_size = 129
        self.hidden_size = 129

        
        #Frequency bin-wise model
        #if n < 1 or n > 127:
        #    self.input_size = 2
        #    self.hidden_size = 1
        #elif (n > 0 and n < 2) or (n < 128 and n > 126):
        #    self.input_size = 3
        #    self.hidden_size = 1
        #elif (n == 2) or (n == 126):
        #    self.input_size = 5
        #    self.hidden_size = 1
        #elif (n == 3) or (n == 125):
        #    self.input_size = 7
        #    self.hidden_size = 1
        #else:
        #    self.input_size = 9
        #    self.hidden_size = 1

        #Single frequency bin
        #self.input_size = 1
        #self.hidden_size = 1

        
        
        self.output_size = self.hidden_size


        self.num_layers = para.num_layers
        self.seq_len = para.seq_len

        super(GRU_REG, self).__init__()

        self.gru = nn.GRU(self.input_size, self.hidden_size, self.num_layers)
        
        self.softplus = nn.Softplus()


    def forward(self, x):

        
        x, h_t = self.gru(x)

        x = self.softplus(x)


        return x


