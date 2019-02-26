# encoding=utf8

import re
import pickle
import itertools
from collections import OrderedDict
import os
import tensorflow as tf
from utils import *
from SeqLabeling import SeqLabeling
import numpy as np
from model import Model
# from loader import load_sentences, update_tag_scheme
# from loader import char_mapping, tag_mapping
# from loader import augment_with_pretrained, prepare_dataset
# from utils import get_logger, make_path, clean, create_model, save_model
# from utils import print_config, save_config, load_config, test_ner
# from data_utils import load_word2vec, create_input, input_from_line, BatchManager

from Arguments import Arguments

flags = tf.app.flags
flags.DEFINE_boolean("clean",       True,      "clean train folder")
flags.DEFINE_boolean("train",       True,      "Whether train the model")
# configurations for the model
flags.DEFINE_integer("seg_dim",     20,         "Embedding size for segmentation, 0 if not used")
flags.DEFINE_integer("char_dim",    100,        "Embedding size for characters")
flags.DEFINE_integer("lstm_dim",    100,        "Num of hidden units in LSTM, or num of filters in IDCNN")
flags.DEFINE_string("tag_schema",   "iob",    "tagging schema iobes or iob")

# configurations for training
flags.DEFINE_float("clip",          5,          "Gradient clip")
flags.DEFINE_float("bias",          10,         "bias Objective function, 0 means no bias")
flags.DEFINE_float("dropout",       0.5,        "Dropout rate")
flags.DEFINE_float("batch_size",    20,         "batch size")
flags.DEFINE_float("lr",            0.001,      "Initial learning rate")
flags.DEFINE_string("optimizer",    "adam",     "Optimizer for training")
flags.DEFINE_boolean("pre_emb",     True,       "Wither use pre-trained embedding")
flags.DEFINE_boolean("zeros",       False,      "Wither replace digits with zero")
flags.DEFINE_boolean("lower",       True,       "Wither lower case")
flags.DEFINE_string("decode_method",       "crf",       "method of decode ,can be crf or lstm")

flags.DEFINE_integer("max_epoch",   100,        "maximum training epochs")
flags.DEFINE_integer("steps_check", 100,        "steps per checkpoint")
flags.DEFINE_string("ckpt_path",    "ckpt",      "Path to save model")
flags.DEFINE_string("summary_path", "summary",      "Path to store summaries")
flags.DEFINE_string("log_file",     "train.log",    "File for log")
flags.DEFINE_string("map_file",     "maps.pkl",     "file for maps")
flags.DEFINE_string("vocab_file",   "vocab.json",   "File for vocab")
flags.DEFINE_string("config_file",  "config_file",  "File for config")
flags.DEFINE_string("script",       "conlleval",    "evaluation script")
flags.DEFINE_string("result_path",  "result",       "Path for results")
flags.DEFINE_string("name",  "combine",       "model name")
# flags.DEFINE_string("emb_file",     os.path.join("data", "vec.txt"),  "Path for pre_trained embedding")
flags.DEFINE_string("emb_file", "../embedding/wiki_100.utf8",  "Path for pre_trained embedding")
# flags.DEFINE_string("train_file",os.path.join("data", "example.train"),  "Path for train data")
flags.DEFINE_string("train_file",   "../combine/combine.train",  "Path for train data")
# flags.DEFINE_string("dev_file",     os.path.join("data", "example.dev"),    "Path for dev data")
flags.DEFINE_string("dev_file",     "../combine/combine.dev",    "Path for dev data")
# flags.DEFINE_string("test_file",    os.path.join("data", "example.test"),   "Path for test data")
flags.DEFINE_string("test_file",    "../combine/combine.test",   "Path for test data")

# flags.DEFINE_string("model_type", "idcnn", "Model type, can be idcnn or bilstm")
flags.DEFINE_string("model_type", "bilstm", "Model type, can be idcnn or bilstm")
flags.DEFINE_string("task_type", "NER", "Task type, can be NER or WordSeg or POS or Rel")

FLAGS = tf.app.flags.FLAGS
assert FLAGS.clip < 5.1, "gradient clip should't be too much"
assert 0 <= FLAGS.dropout < 1, "dropout rate between 0 and 1"
assert FLAGS.lr > 0, "learning rate must larger than zero"
assert FLAGS.decode_method in ["crf", "lstm"]
assert FLAGS.optimizer in ["adam", "sgd", "adagrad","rmsprop"]
assert FLAGS.task_type in ["NER", "WordSeg", "POS", "Rel"]


# FLAGS.ckpt_path = FLAGS.task_type+'/'+FLAGS.ckpt_path
# FLAGS.log_file = FLAGS.task_type+'/'+FLAGS.log_file
# FLAGS.config_file = FLAGS.task_type+'/'+FLAGS.config_file
# FLAGS.map_file = FLAGS.task_type+'/'+FLAGS.map_file



def main(_):
    args = Arguments()
    args.tf_apps(FLAGS)
    ner = SeqLabeling(args)
    if FLAGS.train:
        if FLAGS.clean:
            clean(args)
        ner.train()
    else:
        ner.restore_model()
        with open('/home/shaohui/project/label_sequence/origin_data/opinion.txt') as r,open('../result_data/opinion.Rel.train','w',encoding='utf8') as w:
            for sentence in r.read().split('\n'):
                sentence = re.sub("(?:&.+?;)","",sentence)
                if len(sentence)<2:
                    continue
                print(sentence)
                result = ner.evaluate_line(sentence.strip(),'origin')
                print(result)
                for word,tag in zip(sentence,result):
                    w.write(word+' '+tag+'\n')
                w.write('\n')
                # w.write(' '.join(result)+'\n')
        while True:
            sentence = input('请输入：')
            sentence = re.sub("(?:&.+?;)", "", sentence)
            if len(sentence)<1:
                print([])
            else:
                print(ner.evaluate_line(sentence,args.task_type))


if __name__ == "__main__":
    tf.app.run(main)



