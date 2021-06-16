from transformers import *
bertTok=BertTokenizer.from_pretrained('hfl/chinese-bert-wwm-ext')
bert=BertForSequenceClassification.from_pretrained('hfl/chinese-bert-wwm-ext')