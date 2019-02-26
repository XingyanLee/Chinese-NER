import os
import json
import shutil
import logging
import codecs
import pickle
import itertools
from collections import OrderedDict

import tensorflow as tf

path = os.path.dirname(os.path.realpath(__file__))+'/..'

class Arguments(object):
    def __init__(self):
        self.clean = False
        self.train = False
        self.seg_dim = 20
        self.char_dim = 100
        self.lstm_dim = 100
        self.tag_schema = 'iob'
        self.clip = 5
        self.bias = 10
        self.dropout = 0.5
        self.batch_size = 20
        self.lr = 0.001
        self.optimizer = 'adam'
        self.decode_method = 'crf'
        self.pre_emb = True
        self.zeros = False
        self.lower = True
        self.max_epoch = 100
        self.steps_check = 100
        self.ckpt_path = path+'/ckpt'
        self.log_file = 'log'
        self.map_file = 'maps'
        self.vocab_file = ''
        self.config_file = 'config'
        self.script = 'conlleval'
        self.result_path = 'result'
        self.summary_path = 'summary'
        self.emb_file = path+'/embedding/wiki_100.utf8'
        self.train_file = ''
        self.dev_file = ''
        self.test_file = ''
        self.model_type = 'bilstm'
        self.task_type = 'ner'
        self.name = 'temp'


    def tf_apps(self,FLAGS):
        for k in FLAGS.__dict__['__flags'].keys():
            if k in ['train_file','dev_file','test_file'] and FLAGS.__dict__['__flags'][k]:
                self.__dict__[k] = path+'/data/'+FLAGS.task_type+'/'+FLAGS.name+'/'+FLAGS.__dict__['__flags'][k]
            elif k in [ 'log_file', 'map_file', 'config_file','result_path'] and FLAGS.__dict__['__flags'][k]:
                self.__dict__[k] = path + '/'+self.__dict__[k]+'/' + FLAGS.task_type + '_' + FLAGS.name + '_' + \
                                   FLAGS.__dict__['__flags'][k]
            elif k in ['ckpt_path','summary_path'] and FLAGS.__dict__['__flags'][k]:
                self.__dict__[k] = path + '/'+FLAGS.__dict__['__flags'][k]+'/' + FLAGS.task_type + '/' + FLAGS.name+'/model'
            elif k in self.__dict__.keys() and FLAGS.__dict__['__flags'][k]:
                self.__dict__[k] = FLAGS.__dict__['__flags'][k]

    def dicts(self,dicts):
        for k in dicts.keys():
            if k in ['train_file','dev_file','test_file'] and dicts[k]:
                self.__dict__[k] = path+'/data/'+dicts['task_type']+'/'+dicts['name']+'/'+dicts[k]
            elif k in ['log_file', 'map_file', 'config_file', 'result'] and dicts[k]:
                self.__dict__[k] = path + '/' + self.__dict__[k] + '/' + dicts['task_type'] + '_' + dicts['name'] + '_' + \
                                   dicts[k]
            elif k in ['ckpt_path', 'summary_path'] and dicts[k]:
                self.__dict__[k] = path + '/' + dicts[k] + '/' + dicts['task_type'] + '/' + dicts['name']+'/model'
            elif dicts[k]:
                self.__dict__[k] = dicts[k]
        
    # config for the model
    def config_model(self,char_to_id,tag_to_id):
        self.config = OrderedDict()
        self.config["model_type"] = self.model_type
        self.config["num_chars"] = len(char_to_id)
        self.config["char_dim"] = self.char_dim
        self.config["num_tags"] = len(tag_to_id)
        self.config["seg_dim"] = self.seg_dim
        self.config["lstm_dim"] = self.lstm_dim
        self.config["decode_method"] = self.decode_method
        self.config["batch_size"] = self.batch_size
        self.config["bias"] = self.bias

        self.config["emb_file"] = self.emb_file
        self.config["clip"] = self.clip
        self.config["dropout_keep"] = 1.0 - self.dropout
        self.config["optimizer"] = self.optimizer
        self.config["lr"] = self.lr
        self.config["tag_schema"] = self.tag_schema
        self.config["pre_emb"] = self.pre_emb
        self.config["zeros"] = self.zeros
        self.config["lower"] = self.lower
        return self.config


    def print_config(self,logger):
        """
        Print configuration of the model
        """
        for k, v in self.config.items():
            logger.info("{}:\t{}".format(k.ljust(15), v))

    def save_config(self):
        """
        Save configuration of the model
        parameters are stored in json format
        """
        with open(self.config_file, "w", encoding="utf8") as f:
            json.dump(self.config, f, ensure_ascii=False, indent=4)
        print(self.config)

    def load_config(self, config_file):
        """
        Load configuration of the model
        parameters are stored in json format
        """
        with open(config_file, encoding="utf8") as f:
            self.config = json.load(f)

