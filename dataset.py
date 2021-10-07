import torch
import torch.utils.data as data
import numpy as np
import pandas as pd
import os
import random


class CTDataset(data.Dataset):
    def __init__(self, phase):
        
        self.phase = phase
        csv_file = pd.read_csv('../input/task-2/morbidity.csv')
        patient = np.array(csv_file['Patient'])
        morbidity = np.array(csv_file['Morbidity'])
        
        if self.phase=='train':
            self.Ids = patient[:938]
            self.labels = morbidity[:938]
        elif self.phase=='eval':
            self.Ids = patient[938:1206]
            self.labels = morbidity[938:1206]
        else:
            self.Ids = patient[1206:]
            self.labels = morbidity[1206:]
        

    def __getitem__(self, index):
        
        if self.phase=='train':
            index = random.randint(0,len(self.labels)-1)
        Id = self.Ids[index]
        
        folder_path = '../input/task-2/Covid-19 CT/Covid-19 CT/train/Patient ' + str(Id) + '/CT'
        if index==991:
            # a folder has special naming
            folder_path = '../input/task-2/Covid-19 CT/Covid-19 CT/train/Patient ' + str(Id) + '/2020_1_22'
        all_imgs = list(sorted(os.listdir(folder_path)))
        processed = np.stack([np.array(Image.open(folder_path + '/' + file).resize((256,256)))/255 for file in all_imgs])

        CT = torch.FloatTensor(processed)
        label = torch.FloatTensor([self.labels[index]])
        return (CT, label)

    
    def __len__(self):
        return len(self.labels)