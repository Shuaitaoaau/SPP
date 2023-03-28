#!/usr/bin/env python
# coding=utf-8
# Author: Shuai Tao
# Mail: stao@create.aau.dk
# Create Time: Tue 21 Jun 2022 01:40:08 PM CEST
import os.path

import torch
import numpy as np
import librosa
import soundfile as sf
import gc


def STFT(data):
    spec = librosa.stft(data,
                        n_fft=256,
                        hop_length=128,
                        win_length=256,
                        window='hanning',
                        center=False,
                        )
    mag = np.abs(spec)
    del spec
    gc.collect()
    LPS = np.log(mag ** 2 + 1e-32)
    return LPS.T


def Eval_model(noisy_file, data_mean, data_std):
    noisy_data, fs = sf.read(noisy_file)
    noisy_frames = STFT(noisy_data)


    noisy_frames = torch.from_numpy(noisy_frames).float()
    noisy_frames = (noisy_frames - data_mean) / data_std
    # noisy_frames = torch.unsqueeze(noisy_frames, 0)


    #Loading Model
    model_path = '/user/create.aau.dk/zd77db/SPP/GRU_SPP/model_save'
    for m in range(129):
        model_name = 'model_' + str(m) + '.pth'
        model_name = os.path.join(model_path, model_name)
        trained_model = torch.load(model_name, map_location=torch.device('cpu')).float()
        trained_model.eval()

        with torch.no_grad():    
            
            #val_inputs = noisy_frames[:, m * 1 : (m + 1) * 1]
            #val_inputs = noisy_frames
            #val_inputs = torch.unsqueeze(val_inputs, 0)
            #output = trained_model(val_inputs)
            
            if m < 1:
                val_inputs = noisy_frames[:, m : (m + 2)]
                val_inputs = torch.unsqueeze(val_inputs, 0)
            elif m > 127:
                val_inputs = noisy_frames[:, (m - 1) : (m + 1)]
                val_inputs = torch.unsqueeze(val_inputs, 0)
                output = trained_model(val_inputs)
            elif (m > 0 and m < 2) or (m < 128 and m > 126):
                val_inputs = noisy_frames[:, (m - 1) : (m + 2)]
                val_inputs = torch.unsqueeze(val_inputs, 0)
                output = trained_model(val_inputs)
            #elif (m == 2) or (m == 126):
            #    val_inputs = noisy_frames[:, (m - 2) : (m + 3)]
            #   val_inputs = torch.unsqueeze(val_inputs, 0)
            #    output = trained_model(val_inputs)
            #elif (m == 3) or (m == 125):
            #    val_inputs = noisy_frames[:, (m - 3) : (m + 4)]
            #    val_inputs = torch.unsqueeze(val_inputs, 0)
            #    output = trained_model(val_inputs)
            else:
                val_inputs = noisy_frames[:, (m - 2) : (m + 3)]
                val_inputs = torch.unsqueeze(val_inputs, 0)
 

            
            #val_inputs = noisy_frames[:, m * 3 : (m + 1) * 3]
            #val_inputs = torch.unsqueeze(val_inputs, 0)

            #val_inputs = noisy_frames[:, m * 1 : (m + 1) * 1]
            #val_inputs = torch.unsqueeze(val_inputs, 0)


            #if m < 1 or m > 127:
            #    val_inputs = noisy_frames[:, m : m + 1]
            #    val_inputs = torch.unsqueeze(val_inputs, 0)
            #else:
            #    val_inputs = noisy_frames[:, m - 1 : m + 2]
            #    val_inputs = torch.unsqueeze(val_inputs, 0)
            
            #full band
            #val_inputs = noisy_frames
            #val_inputs = torch.unsqueeze(val_inputs, 0)
 
            output = trained_model(val_inputs)

        if m == 0:
            temp1 = output
        else:
            temp1 = torch.cat((temp1, output), 2)

    temp1 = temp1.transpose(2, 1)


    return temp1


if __name__ == "__main__":

    data_mean = torch.load('/user/create.aau.dk/zd77db/SPP/Encoder_test5/Data/total_mean.pt')
    data_std = torch.load('/user/create.aau.dk/zd77db/SPP/Encoder_test5/Data/total_std.pt')
    test_file_path = '/user/create.aau.dk/zd77db/test_set/test.csv'
    test_file = np.loadtxt(test_file_path, dtype='str')
    noisy_file = test_file[:, 0].tolist()
    file_len = len(noisy_file)

    for i in range(file_len):
        test_output = Eval_model(noisy_file[i], data_mean, data_std)
        if i == 0:
            temp = test_output
        else:
            temp = torch.cat((temp, test_output), 0)

        print(temp.shape)

    torch.save(temp, '/user/create.aau.dk/zd77db/SPP/comparison/test/sub_2.pt')

