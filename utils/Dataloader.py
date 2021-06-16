import logging

class Instance():
    def __init__(self):
        self.sentence = None
        self.labels=None

class Vocab():
    def __init__(self) -> None:
        
        self.labels=['PAD','UNK']
        self.labels2id={}
    
    def build(self,data):
        """构建词表

        :param data: 训练数据 List，元素类型为 Instance
        :type data: list
        """        
        label=set()
        for inst in data:
            label=label|set(inst.ner_labels)
        self.labels.extend(sorted(list(label)))
        
        for idx,v in enumerate(self.BIO_vocab):
            self.label2id[v]=[idx]
        assert len(self.labels)==len(self.labels2id)
        logging.info('labels num:{}')
        for k ,v in self.labels2id.items():
            logging.info('{}\t{}'.format(v,k))


