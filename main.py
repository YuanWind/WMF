from modules.Model import Model
from utils.Dataloader import Vocab
from utils.Trainer import eval_dev, train
import logging
from utils.utils import load_pkl, set_logger
import os
import argparse
from Config import Config


def set_args():
    argparser=argparse.ArgumentParser()
    argparser.add_argument('--config_file',default='config.cfg')
    argparser.add_argument('--use_cuda',action='store_true', default=True)
    argparser.add_argument('--is_train',action='store_true', default=True)
    argparser.add_argument('--log_file', default='tmp/log.log')
    args,extra_args=argparser.parse_known_args()
    set_logger(args.log_file,to_console=True,to_file=False)
    logging.info("Process ID {}, Process Parent ID {}, start...".format(os.getpid(), os.getppid()))
    logging.info('Your args:')
    for k,v in vars(args).items():
        logging.info('{}={}'.format(k,v))
    return args,extra_args

    
if __name__ == '__main__':
    args,extra_args=set_args()
    
    config=Config(args.config_file,extra_args)
    
    train_data=load_pkl(config.train_file) # list类型
    dev_data=load_pkl(config.dev_file)     # list 类型
    
    vocab=Vocab()
    vocab.build(train_data)
    
    model=Model(config,vocab)
    model.build()

    if args.is_train:
        train(train_data,dev_data,model,config)
    eval_dev(dev_data,model,config)
    