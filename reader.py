import os,sys
from sklearn.model_selection import train_test_split
from collections import defaultdict, Counter
from label_dictionary import LabelDictionary
import nltk
import numpy as np
import pdb
from sklearn.preprocessing import MultiLabelBinarizer


UNK_TOKEN = "<UNK>"

class Reader:
  def __init__(self,filename):
    np.random.seed(42)
    self.label_set = set()
    lines = open(filename,'r').read().strip('\n').split('\n')
    lines = [x.split('\t') for x in lines if x.strip(" ")!='']
    train,test = train_test_split(lines,test_size=800,random_state=42)
    
    text_train,y_train = self.split_instance(train,add_label=True)
    text_test,y_test = self.split_instance(test)
    self.vocab_thr = 1
    self.vocab = self.get_vocab(text_train)
    self.train_txt = self.filter_data(text_train)
    self.test_txt = self.filter_data(text_test)

    self.label_bin = MultiLabelBinarizer()
    self.label_bin.fit(y_train)

    self.train_txt,self.train_labels = self.shuffle_data(self.train_txt,y_train)
    self.test_txt,self.test_labels = self.shuffle_data(self.test_txt,y_test)
    self.train_labels = self.label_bin.transform(self.train_labels)
    self.test_labels = self.label_bin.transform(self.test_labels)


  def shuffle_data(self,x,y):
    ids = np.arange(len(x))
    np.random.shuffle(ids)
    xx = [x[i] for i in ids]
    yy = [y[i] for i in ids]
    return xx,yy

  def get_vocab(self,data):
    vocab = Counter()
    for text in data:
      tokens = [x.lower()  for x in nltk.word_tokenize(text)]
      vocab.update(tokens)

    toks = list(vocab.keys())
    for tok in toks:
      if vocab[tok] <= self.vocab_thr:
        del vocab[tok]
    #
    return vocab

  def get_label_names(self):
    return self.label_bin.classes_

  def split_instance(self,data,add_label=False):
    docs = []
    labels = []
    for text,label_line in data:
      x = text.lower()
      labs = [y for y in label_line.lower().split(",") if y!='']
      if add_label:
        self.label_set.update(labs)
      else:
        labs = [x for x in labs if x in self.label_set]
      if len(labs)==0:
        continue
      docs.append(x)
      labels.append(labs)
    #
  
    return docs,labels



  def filter_data(self,data):
    str_list = []
    for text in data:
      tokens = [x.lower()  for x in nltk.word_tokenize(text)]
      tokens = [x if x in self.vocab else UNK_TOKEN for x in tokens]
      str_list.append(" ".join(tokens))
      # str_list.append(tokens)

    return str_list
