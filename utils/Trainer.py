import logging
import time

from torch.utils.data.dataloader import DataLoader
from modules.Optimizer import Optimizer
import torch
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp.grad_scaler import GradScaler
def train(train_data,dev_data,model,config):
    
    train_loader=DataLoader(train_data,config.train_batch_size,config.shuffle,collate_fn=model.my_collate_fn)
    optimizer=Optimizer(model.parameters(),config,'Adam')
    scaler=GradScaler()
    best_score=0
    global_step=0
    total_batch=int(len(train_data)/config.train_batch_size)
    for iter in range(config.train_iters):
        start_time=time.time()
        logging.info('Current Iteration:{}'.format(iter))
        cur_batch=0
        
        for onebatch in train_loader:
            # TODO dataloader
            inputs,labels=onebatch
            model.train()
            with autocast():
                model.forward(inputs)
                loss=model.compute_loss(inputs)
                loss=loss/config.update_every
                
            loss_value=loss.data.cpu().numpy()
            scaler.scale(loss).backward()
            during_time = float(time.time() - start_time)
            # TODO 计算acc or F1
            acc=0
            logging.info('step:%d, iter:%d, batch:%d, time:%d, loss:%.3f, acc:%.3f;'\
                %(global_step, iter, cur_batch, during_time, loss_value, acc))
            
            cur_batch+=1
            if cur_batch%config.update_every==0 or cur_batch==total_batch:
                scaler.unscale_(optimizer.optim)
                model.clip_grad_norm_()
                scaler.step(optimizer.optim)
                scaler.update()
                optimizer.schedule()

                optimizer.zero_grad()
                global_step += 1
            if cur_batch % config.validate_every == 0 or cur_batch == total_batch:
                score=eval_dev(dev_data,model,config)
                if score>best_score:
                    logging.info("Exceed best score: history = %.2f, current = %.2f" %(best_score,score))
                    best_score=score
                    if config.save_after>=0 and iter>config.save_after:
                        torch.save(model.state_dict(),config.model_file)
                        logging.info('Save model to: {}'.format(config.model_file))
                        
def eval_dev(dev_data,model,config,load_model=False):
    
    dev_loader=DataLoader(dev_data,config.test_batch_size,shuffle=False,collate_fn=model.my_collate_fn)
    if load_model:
        model.load_state_dict(torch.load(config.model_file))
    for onebatch in dev_loader:
        pass
    
    score=0
    logging.info('Evaluate on dev: ... ')
    
    return score
                            