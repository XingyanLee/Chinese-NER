# encoding=utf8
import os
import pickle
from collections import OrderedDict

from Helper import *
from model import Model
from utils import *
import tensorflow as tf


class SeqLabeling(object):
    def __init__(self,args):
        self.graph = tf.Graph()
        self.args = args
        make_path(self.args)
        self.logger = get_logger(self.args.log_file)


    def create_helper(self,helper=None):
        if helper:
            self.helper = helper
            return
        self.helper = Helper()
        self.helper.load_train(self.args.train_file, self.args.lower, self.args.zeros)
        self.helper.load_dev(self.args.dev_file, self.args.lower, self.args.zeros)
        self.helper.load_test(self.args.test_file, self.args.lower, self.args.zeros)

        # self.helper.select_tag_schema(self.args.tag_schema)

        self.helper.load_pretrained_emb(self.args.emb_file)

        self.helper.create_maps(self.args.map_file, self.args.lower)

        self.helper.create_batch(self.args.batch_size)

        # make path for store log and model if not exist
        if os.path.isfile(self.args.config_file):
            self.args.load_config(self.args.config_file)
        else:
            self.args.config_model(self.helper.char_to_id, self.helper.tag_to_id)

            self.args.save_config()



        self.args.print_config(self.logger)

        self.model = None

    def save_model(self):
        self.model.saver.save(self.sess, self.args.ckpt_path)
        self.logger.info("model saved")

    def create_model(self):
        # create model, reuse parameters if exists
        self.model = Model(self.args.config,graph = self.graph)
        ckpt = tf.train.get_checkpoint_state(self.args.ckpt_path[:-6])
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            self.logger.info("Reading model parameters from %s" % ckpt.model_checkpoint_path)
            self.model.saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            self.logger.info("Created model with fresh parameters.")
            self.sess.run(tf.global_variables_initializer())
            if self.args.config["pre_emb"]:
                emb_weights = self.sess.run(self.model.char_lookup.read_value())
                emb_weights = DataUtils.load_word2vec(self.args.config["emb_file"], self.helper.id_to_char, self.args.config["char_dim"], emb_weights)
                self.sess.run(self.model.char_lookup.assign(emb_weights))
                self.logger.info("Load pre-trained embedding.")

    def train(self):
        self.create_helper()
        # limit GPU memory
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        steps_per_epoch = self.helper.train_batch.len_data
        with tf.Session(config=tf_config,graph=self.graph) as self.sess:
            self.create_model()
            self.logger.info("start training")
            loss = []
            self.train_summary = tf.summary.FileWriter(self.args.summary_path + '/train',self.sess.graph)
            self.dev_summary = tf.summary.FileWriter(self.args.summary_path + '/dev',self.sess.graph)
            for i in range(self.args.max_epoch):
                print("epoch"+str(i))
                for batch in self.helper.train_batch.iter_batch(shuffle=True):
                    self.step, batch_loss, summary = self.model.run_step(self.sess, 'train', batch)

                    loss.append(batch_loss)
                    if self.step % self.args.steps_check == 0:
                        iteration = self.step // steps_per_epoch + 1
                        self.train_summary.add_summary(summary, self.step)
                        self.logger.info("iteration:{} step:{}/{}, "
                                    "{} loss:{:>9.6f}".format(
                            iteration,  self.step % steps_per_epoch, steps_per_epoch, self.args.task_type, np.mean(loss)))
                        loss = []

                best = self.evaluate("dev")
                if best:
                    self.save_model()
                self.evaluate("test")

    def evaluate(self, name):
        self.logger.info("evaluate:{}".format(name))
        ner_results,losses = self.model.evaluate(self.sess, self.helper.__dict__[name+'_batch'], self.helper.id_to_tag, name)
        eval_lines = test_ner(ner_results, self.args.result_path)
        for line in eval_lines:
            self.logger.info(line)
        f1 = float(eval_lines[1].strip().split()[-1])
        if name == "dev":
            summary = tf.Summary(value=[
                tf.Summary.Value(tag="dev_loss", simple_value=np.mean(losses)),
            ])
            self.dev_summary.add_summary(summary, self.step)
            best_test_f1 = self.model.best_dev_f1.eval()
            if f1 > best_test_f1:
                tf.assign(self.model.best_dev_f1, f1).eval()
                self.logger.info("new best dev f1 score:{:>.3f}".format(f1))
            return f1 > best_test_f1
        elif name == "test":
            best_test_f1 = self.model.best_test_f1.eval()
            if f1 > best_test_f1:
                tf.assign(self.model.best_test_f1, f1).eval()
                self.logger.info("new best test f1 score:{:>.3f}".format(f1))
            return f1 > best_test_f1

    def restore_model(self):
        self.args.load_config(self.args.config_file)
        # limit GPU memory
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        with open(self.args.map_file, "rb") as f:
            self.char_to_id, self.id_to_char, self.tag_to_id, self.id_to_tag = pickle.load(f)
        # print(char_to_id,id_to_char,tag_to_id)
        self.sess = tf.Session(config=tf_config,graph=self.graph)
        self.create_model()
    def evaluate_line(self,sentence,rettype='NER'):
        tags = self.model.evaluate_line(self.sess, DataUtils.input_from_line(sentence, self.char_to_id), self.id_to_tag)
        if rettype=='origin':
            return tags
        elif rettype=='NER':
            return extract(sentence,tags)
        elif rettype=='WordSeg':
            result = extract(sentence,tags)
            wordseg_result = [sentence[item['start']:item['end']] for item in result]
            return wordseg_result
        elif rettype=='POS':
            result =extract(sentence,tags)
            pos_result = [sentence[item['start']:item['end']]+'/'+item['type'] for item in result]
            return pos_result
        elif rettype=='Rel':
            result = extract(sentence,tags)
            # wordseg_result = [sentence[item['start']:item['end']] for item in result]
            return result