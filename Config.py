from configparser import ConfigParser, ExtendedInterpolation
import os
import logging
class Config:
    def __init__(self,config_file,extra_args=None) -> None:
        config=ConfigParser(interpolation=ExtendedInterpolation())
        config.read(config_file)
        if extra_args:
            extra_args=vars(extra_args)
            for section in config.sections():
                for k,v in config.items(section):
                    if k in extra_args:
                        v=type(v)(extra_args[k])
                        config.set(section,k,v)
        self._config=config
        
        self.set_property()
        
        if not os.path.isdir(self.save_dir):
            os.mkdir(self.save_dir)
            
        for section in config.sections():
            for k, v in config.items(section):
                logging.info('{}={}'.format(k,v))
            
    def set_property(self):
        # Setting Section
        self.seed=self._config.getint('Setting','seed')
        self.train_iters=self._config.getint('Setting','train_iters')
        self.train_batch_size=self._config.getint('Setting','train_batch_size')
        self.shuffle=self._config.getboolean('Setting','shuffle')
        self.update_every=self._config.getint('Setting','update_every')
        self.test_batch_size=self._config.getint('Setting','test_batch_size')
        self.validate_steps=self._config.getint('Setting','validate_steps')
        self.save_after=self._config.getint('Setting','save_after')
        
        # Data Section
        self.data_dir=self._config.get('Data','data_dir')
        self.train_file=self._config.get('Data','train_file')
        self.dev_file=self._config.get('Data','dev_file')
        
        # Save Section
        self.save_dir=self._config.get('Save','save_dir')
        self.model_file=self._config.get('Save','model_file')
        
        # Network Section
        self.max_len=self._config.getint('Network','max_len')
        
        #Optimizer Section
        self.learning_rate=self._config.getfloat('Optimizer','learning_rate')
        self.beta_1=self._config.getfloat('Optimizer','beta_1')
        self.beta_2=self._config.getfloat('Optimizer','beta_2')
        self.decay=self._config.getfloat('Optimizer','decay')
        self.decay_steps=self._config.getint('Optimizer','decay_steps')
        self.epsilon=self._config.getfloat('Optimizer','epsilon')
        self.clip=self._config.getfloat('Optimizer','clip')