import torch
import torch.nn as nn

class Model():
    def __init__(self,config,vocab):
        self.config=config
        self.vocab=vocab
        
    def build(self):
        pass
    
    def train(self):
        pass
    
    def eval(self):
        pass
    
    def my_collate_fn(self,data):
        """

        Args:
            data (list): 一个batch的list数据
        return：
            一个batch的模型输入
        """
        pass
    
    def parameters(self):
        pass
    
    def state_dict(self):
        pass
    
    def load_state_dict(self,params):
        pass
    
    def forward(self):
        pass
    
    def clip_grad_norm_(self):
        nn.utils.clip_grad_norm_(param, max_norm=self.config.clip)
        
    def compute_loss(self,labels,labels_mask):
        pass
    def compute_acc(self,labels,labels_mask):
        pass