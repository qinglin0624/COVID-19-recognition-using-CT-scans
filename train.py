import torch
import numpy as np
import time
import json
import warnings
warnings.filterwarnings("ignore")


def calculate_step(set_size, dot_per_epoch):
    num_iter = set_size / BATCH_SIZE
    step = num_iter / dot_per_epoch
    return int(step)


def train_model(model, optimizer, train_loader, valid_loader, criterion, EPOCH):
    
    train_begin = time.time()

    for epoch in range(EPOCH):
        print('Epoch {}/{}'.format(epoch, EPOCH - 1))
        print('-' * 10)
        
        model.train()        
        ptime=time.time()
        
        train_loss, train_correct = 0,0
        
        for itera, (x, y) in enumerate(train_loader):
            itime=time.time()
            x,  y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            optimizer.zero_grad()
            preds = model(x)
            
            loss = criterion(preds,y)

            loss.backward()
            optimizer.step()
            
            preds_o = torch.argsort(-preds, axis=1)
            c = (preds_o[:,0]==y).sum().item()
            train_correct += c
            train_loss += loss.item() *  y.size(0)
            print(y.size(0))
            
            print("Iteration: {:6}  loss: {:.5f}  model time: {:.3f}s iter time: {:.3f}s"\
                  .format(itera,loss.item(),time.time()-itime,time.time()-ptime))
            ptime =time.time()
            
            if itera%STEP==0 and itera//STEP>0:
                
                with torch.no_grad():
                    model.eval()
                    val_loss, val_correct = 0,0

                    for batch_idx, (x, y) in enumerate(valid_loader):
                        x,  y = x.cuda(non_blocking=True),  y.cuda(non_blocking=True)
                        preds = model(x)

                        loss = criterion(preds,y)

                        val_loss += loss.item()* y.size(0)
                        preds_o = torch.argsort(-preds, axis=1)
                        val_correct += (preds_o[:,0]==y).sum().item()

                    val_loss = val_loss / valid_size
                    val_acc = val_correct / valid_size
                    stats['valid'][0].append(val_loss)
                    stats['valid'][1].append(val_acc)
                    print ('val loss: {:.4f}   acc: {:.4f}'.format(val_loss, val_acc))
                
                model.train()

        epoch_loss = train_loss / train_size
        epoch_acc = train_correct / train_size
        stats['train'][0].append(epoch_loss)
        stats['train'][1].append(epoch_acc)
        print ('train loss: {:.4f}   acc: {:.4f}'.format(epoch_loss, epoch_acc))
                   
        json.dump(stats, open(TRIAL + '_stats.json','w'))
        torch.save(model.state_dict(), TRIAL+ '_'+ str(epoch)+'e.pth')

    train_length = time.time() - train_begin
    print('Training complete in {:.0f}m {:.0f}s'.format(train_length // 60, train_length % 60))
