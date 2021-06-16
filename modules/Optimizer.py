import torch
class Optimizer:
    def __init__(self, parameter, config, optimizer='Adam'):
        self.parameter=parameter
        self.config=config
        if optimizer=='Adam':
            self.init_Adam()
    def init_Adam(self):
        self.optim = torch.optim.Adam(self.parameter, 
                                      lr=self.config.learning_rate, 
                                      betas=(self.config.beta_1, self.config.beta_2),
                                      eps=self.config.epsilon)
        decay, decay_step =self.config.decay,self.config.decay_steps
        l = lambda epoch: decay ** (epoch // decay_step)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optim, lr_lambda=l)
    
    def step(self):
        self.optim.step()
        self.scheduler.step()
        self.optim.zero_grad()

    def zero_grad(self):
        self.optim.zero_grad()

    @property
    def lr(self):
        return self.optim.state_dict()['param_groups'][0]['lr']