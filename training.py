from train import train_model, calculate_step
from dataset import CTDataset
from model import ResModel
import torch
from torch.utils.data import DataLoader
import json


    
if __name__=='__main__':
    LR = 1e-3
    EPOCH = 10
    BATCH_SIZE=8
    TRIAL = 'first'
    
    trainset = CTDataset('train')
    validset = CTDataset('val')
    # testset = CTDataset('test')
    train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
    valid_loader = DataLoader(validset, batch_size=BATCH_SIZE, num_workers=8, pin_memory=True)
    # test_loader = DataLoader(testset, batch_size=BATCH_SIZE, num_workers=8, pin_memory=True)
    train_size = len(trainset)
    valid_size = len(validset)
    # test_size = len(testset)

    model = ResModel().cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    stats = {'train':[[],[]], 'valid':[[],[]]}
    
    STEP = calculate_step(set_size=train_size, dot_per_epoch=10)
    train_model(model, optimizer, train_loader, valid_loader, criterion, EPOCH)
    
    json.dump(stats, open('output/'+TRIAL + '_stats.json','w'))
    torch.save(model.state_dict(), 'output/'+TRIAL+ '.pth')
