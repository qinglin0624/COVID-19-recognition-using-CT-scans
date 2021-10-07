import pandas as pd
import numpy as np
import os
import torch
from PIL import Image
import time

def generate_npy(phase, p, l):
  s = time.time()
  for index in range(len(p)):
      Id = p[index]
      label = l[index]
      print(Id)
      folder_path = 'Covid-19 CT/Covid-19 CT/train/Patient ' + str(Id) + '/CT'
      if Id==991:
          # a folder has special naming
          folder_path = 'Covid-19 CT/Covid-19 CT/train/Patient ' + str(Id) + '/2020_1_22'
      try:
          all_imgs = list(sorted(os.listdir(folder_path)))
      except:
          folder_path = folder_path.replace('train','test')
          all_imgs = list(sorted(os.listdir(folder_path)))
      processed = np.stack([np.array(Image.open(folder_path + '/' + file).resize((256,256)))/255 for file in all_imgs])
      CT = torch.FloatTensor(processed)
      CT = CT.permute(0,3,1,2)
      CT = torch.nn.functional.interpolate(CT, size=128, mode="bilinear")        
      CT_ = CT.permute(1,2,3,0)
      CT_ = torch.nn.functional.interpolate(CT_, size=(128,51), mode="bilinear")
      CT_ = CT_.permute(0,3,1,2)
      np.save('Covid-19 CT/'+phase+'/P'+str(Id)+'L'+str(label)+'.npy',np.array(CT_))

  length = time.time() - s
  print('complete in {:.0f}m {:.0f}s'.format(length // 60, length % 60))
  

if __name__=='__main':
  csv_file = pd.read_csv('Covid-19 CT/Covid-19 CT/morbidity.csv')
  patient = np.array(csv_file['Patient'])
  morbidity = np.array(csv_file['Morbidity'])

  train = patient[:938]
  train_l = morbidity[:938]
  val = patient[938:1206]
  val_l = morbidity[938:1206]
  test = patient[1206:]
  test_l = morbidity[1206:]
  
  generate_npy('train', train, train_l)
  generate_npy('val', val, val_l)
  generate_npy('test', test, test_l)
  
