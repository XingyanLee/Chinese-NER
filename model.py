# encoding = utf8
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib.crf import crf_log_likelihood
from tensorflow.contrib.crf import viterbi_decode
from tensorflow.contrib.layers.python.layers import initializers

from utils import *
from Helper import DataUtils


class Model(object):
    def __init__(self, config ,graph):
        with graph.as_default():
            self.config = config

            self.lr = config["lr"]
            self.char_dim = config["char_dim"]
            self.lstm_dim = config["lstm_dim"]
            self.seg_dim = config["seg_dim"]
            self.decode_method = config["decode_method"]
            self.num_tags = config["num_tags"]
            self.num_chars = config["num_chars"]
            self.num_segs = 4
            self.num_heads = 8
            self.bias = config['bias']
            self.num_units = 2*int(config["lstm_dim"])
            self.global_step = tf.Variable(0, trainable=False)
            self.best_dev_f1 = tf.Variable(0.0, trainable=False)
            self.best_test_f1 = tf.Variable(0.0, trainable=False)
            self.initializer = initializers.xavier_initializer()

            self.train_summary = ""
            self.dev_summary = ""

            # add placeholders for the model

            self.char_inputs = tf.placeholder(dtype=tf.int32,
                                              shape=[None, None],
                                              name="ChatInputs")
            self.seg_inputs = tf.placeholder(dtype=tf.int32,
                                             shape=[None, None],
                                             name="SegInputs")

            self.targets = tf.placeholder(dtype=tf.int32,
                                          shape=[None, None],
                                          name="Targets")
            # dropout keep prob
            self.dropout = tf.placeholder(dtype=tf.float32,
                                          name="Dropout")

            used = tf.sign(tf.abs(self.char_inputs))
            length = tf.reduce_sum(used, reduction_indices=1)
            self.lengths = tf.cast(length, tf.int32)
            self.batch_size = tf.shape(self.char_inputs)[0]
            self.num_steps = tf.shape(self.char_inputs)[-1]


            #Add model type by crownpku， bilstm or idcnn
            self.model_type = config['model_type']
            #parameters for idcnn
            self.layers = [
                {
                    'dilation': 1
                },
                {
                    'dilation': 1
                },
                {
                    'dilation': 2
                },
            ]
            self.filter_width = 3
            self.num_filter = self.lstm_dim
            self.embedding_dim = self.char_dim + self.seg_dim
            self.repeat_times = 4
            self.cnn_output_width = 0



            # embeddings for chinese character and segmentation representation
            embedding = self.embedding_layer(self.char_inputs, self.seg_inputs, config)

            if self.model_type == 'bilstm':
                # apply dropout before feed to lstm layer
                model_inputs = tf.nn.dropout(embedding, self.dropout)

                # bi-directional lstm layer
                model_outputs = self.biLSTM_layer(model_inputs, self.lstm_dim, self.lengths)
                # model_outputs = self.self_attention(model_outputs)
                if self.decode_method=='crf':
                    # logits for tags
                    self.logits = self.project_layer_bilstm(model_outputs)
                elif self.decode_method=='lstm':
                    self.logits = self.lstm_decode_layer(model_outputs)

            elif self.model_type == 'idcnn':
                # apply dropout before feed to idcnn layer
                model_inputs = tf.nn.dropout(embedding, self.dropout)

                # ldcnn layer
                model_outputs = self.IDCNN_layer(model_inputs)

                # logits for tags
                self.logits = self.project_layer_idcnn(model_outputs)

            else:
                raise KeyError

            # loss of the model
            if self.decode_method=='crf':
                self.loss = self.loss_layer(self.logits, self.lengths)
            elif self.decode_method=='lstm':
                self.loss = self.lstm_decode_loss_layer(self.logits,self.bias)
            print(self.loss)
            self.summary = tf.summary.scalar('loss', self.loss)

            with tf.variable_scope("optimizer"):
                optimizer = self.config["optimizer"]
                if optimizer == "sgd":
                    self.opt = tf.train.GradientDescentOptimizer(self.lr)
                elif optimizer == "adam":
                    self.opt = tf.train.AdamOptimizer(self.lr)
                elif optimizer == "adgrad":
                    self.opt = tf.train.AdagradOptimizer(self.lr)
                elif optimizer == 'rmsprop':
                    self.opt = tf.train.RMSPropOptimizer(self.lr)
                else:
                    raise KeyError

                # apply grad clip to avoid gradient explosion
                grads_vars = self.opt.compute_gradients(self.loss)
                capped_grads_vars = [[tf.clip_by_value(g, -self.config["clip"], self.config["clip"]), v]
                                     for g, v in grads_vars]
                self.train_op = self.opt.apply_gradients(capped_grads_vars, self.global_step)

            # saver of the model
            self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

    def normalize(self,inputs, epsilon = 1e-8,scope="ln",reuse=None):
        with tf.variable_scope(scope, reuse=reuse):
            inputs_shape = inputs.get_shape()
            params_shape = inputs_shape[-1:]
            mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
            beta = tf.Variable(tf.zeros(params_shape),dtype=tf.float32)
            gamma = tf.Variable(tf.ones(params_shape),dtype=tf.float32)
            normalized = (inputs - mean) / ((variance + epsilon) ** (0.5))
            outputs = gamma * normalized + beta
        return outputs

    def self_attention(self,keys, scope='multihead_attention', reuse=None):
        with tf.variable_scope(scope, reuse=reuse):
            Q = tf.nn.relu(tf.layers.dense(keys, self.num_units, kernel_initializer=tf.contrib.layers.xavier_initializer()))
            K = tf.nn.relu(tf.layers.dense(keys, self.num_units, kernel_initializer=tf.contrib.layers.xavier_initializer()))
            V = tf.nn.relu(tf.layers.dense(keys, self.num_units, kernel_initializer=tf.contrib.layers.xavier_initializer()))
            Q_ = tf.concat(tf.split(Q, self.num_heads, axis=2), axis=0)
            K_ = tf.concat(tf.split(K, self.num_heads, axis=2), axis=0)
            V_ = tf.concat(tf.split(V, self.num_heads, axis=2), axis=0)
            outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))
            outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)
            key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1)))
            key_masks = tf.tile(key_masks, [self.num_heads, 1])
            key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(keys)[1], 1])
            paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
            outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)
            outputs = tf.nn.softmax(outputs)
            query_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1)))
            query_masks = tf.tile(query_masks, [self.num_heads, 1])
            query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]])
            outputs *= query_masks
            outputs = tf.nn.dropout(outputs, keep_prob=self.dropout)
            outputs = tf.matmul(outputs, V_)
            outputs = tf.concat(tf.split(outputs, self.num_heads, axis=0), axis=2)
            outputs += keys
            outputs = self.normalize(outputs)
        return outputs

    def embedding_layer(self, char_inputs, seg_inputs, config, name=None):
        """
        :param char_inputs: one-hot encoding of sentence
        :param seg_inputs: segmentation feature
        :param config: wither use segmentation feature
        :return: [1, num_steps, embedding size], 
        """

        embedding = []
        with tf.variable_scope("char_embedding" if not name else name), tf.device('/cpu:0'):
            self.char_lookup = tf.get_variable(
                    name="char_embedding",
                    shape=[self.num_chars, self.char_dim],
                    initializer=self.initializer)
            embedding.append(tf.nn.embedding_lookup(self.char_lookup, char_inputs))
            if config["seg_dim"]:
                with tf.variable_scope("seg_embedding"), tf.device('/cpu:0'):
                    self.seg_lookup = tf.get_variable(
                        name="seg_embedding",
                        shape=[self.num_segs, self.seg_dim],
                        initializer=self.initializer)
                    embedding.append(tf.nn.embedding_lookup(self.seg_lookup, seg_inputs))
            embed = tf.concat(embedding, axis=-1)
        return embed

    def biLSTM_layer(self, model_inputs, lstm_dim, lengths, name=None):
        """
        :param lstm_inputs: [batch_size, num_steps, emb_size] 
        :return: [batch_size, num_steps, 2*lstm_dim] 
        """
        with tf.variable_scope("char_BiLSTM" if not name else name):
            lstm_cell = {}
            for direction in ["forward", "backward"]:
                with tf.variable_scope(direction):
                    # lstm_cell[direction] = rnn.CoupledInputForgetGateLSTMCell(
                    #     lstm_dim,
                    #     use_peepholes=True,
                    #     initializer=self.initializer,
                    #     state_is_tuple=True)
                    lstm_cell[direction] = rnn.LSTMCell(
                        lstm_dim,
                        initializer=self.initializer,
                        state_is_tuple=True)
            outputs, final_states = tf.nn.bidirectional_dynamic_rnn(
                lstm_cell["forward"],
                lstm_cell["backward"],
                model_inputs,
                dtype=tf.float32,
                sequence_length=lengths)
            # 修改-李星彦
            outputs = tf.concat(outputs, axis=2)
            # outputs = self.self_attention(outputs)
        return outputs
    
    #IDCNN layer 
    def IDCNN_layer(self, model_inputs, 
                    name=None):
        """
        :param idcnn_inputs: [batch_size, num_steps, emb_size] 
        :return: [batch_size, num_steps, cnn_output_width]
        """
        model_inputs = tf.expand_dims(model_inputs, 1)
        reuse = False
        if self.dropout == 1.0:
            reuse = True
        with tf.variable_scope("idcnn" if not name else name):
            shape=[1, self.filter_width, self.embedding_dim,
                       self.num_filter]
            print(shape)
            filter_weights = tf.get_variable(
                "idcnn_filter",
                shape=[1, self.filter_width, self.embedding_dim,
                       self.num_filter],
                initializer=self.initializer)
            
            """
            shape of input = [batch, in_height, in_width, in_channels]
            shape of filter = [filter_height, filter_width, in_channels, out_channels]
            """
            layerInput = tf.nn.conv2d(model_inputs,
                                      filter_weights,
                                      strides=[1, 1, 1, 1],
                                      padding="SAME",
                                      name="init_layer")
            finalOutFromLayers = []
            totalWidthForLastDim = 0
            for j in range(self.repeat_times):
                for i in range(len(self.layers)):
                    dilation = self.layers[i]['dilation']
                    isLast = True if i == (len(self.layers) - 1) else False
                    with tf.variable_scope("atrous-conv-layer-%d" % i,
                                           reuse=True
                                           if (reuse or j > 0) else False):
                        w = tf.get_variable(
                            "filterW",
                            shape=[1, self.filter_width, self.num_filter,
                                   self.num_filter],
                            initializer=tf.contrib.layers.xavier_initializer())
                        b = tf.get_variable("filterB", shape=[self.num_filter])
                        conv = tf.nn.atrous_conv2d(layerInput,
                                                   w,
                                                   rate=dilation,
                                                   padding="SAME")
                        conv = tf.nn.bias_add(conv, b)
                        conv = tf.nn.relu(conv)
                        if isLast:
                            finalOutFromLayers.append(conv)
                            totalWidthForLastDim += self.num_filter
                        layerInput = conv
            finalOut = tf.concat(axis=3, values=finalOutFromLayers)
            keepProb = 1.0 if reuse else 0.5
            finalOut = tf.nn.dropout(finalOut, keepProb)

            finalOut = tf.squeeze(finalOut, [1])
            finalOut = tf.reshape(finalOut, [-1, totalWidthForLastDim])
            self.cnn_output_width = totalWidthForLastDim
            return finalOut

    def project_layer_bilstm(self, lstm_outputs, name=None):
        """
        hidden layer between lstm layer and logits
        :param lstm_outputs: [batch_size, num_steps, emb_size] 
        :return: [batch_size, num_steps, num_tags]
        """
        with tf.variable_scope("project"  if not name else name):
            with tf.variable_scope("hidden"):
                W = tf.get_variable("W", shape=[self.lstm_dim*2, self.lstm_dim],
                                    dtype=tf.float32, initializer=self.initializer)

                b = tf.get_variable("b", shape=[self.lstm_dim], dtype=tf.float32,
                                    initializer=tf.zeros_initializer())
                output = tf.reshape(lstm_outputs, shape=[-1, self.lstm_dim*2])
                hidden = tf.tanh(tf.nn.xw_plus_b(output, W, b))

            # project to score of tags
            with tf.variable_scope("logits"):
                W = tf.get_variable("W", shape=[self.lstm_dim, self.num_tags],
                                    dtype=tf.float32, initializer=self.initializer)

                b = tf.get_variable("b", shape=[self.num_tags], dtype=tf.float32,
                                    initializer=tf.zeros_initializer())

                pred = tf.nn.xw_plus_b(hidden, W, b)

            return tf.reshape(pred, [-1, self.num_steps, self.num_tags])
    
    #Project layer for idcnn by crownpku
    #Delete the hidden layer, and change bias initializer
    def project_layer_idcnn(self, idcnn_outputs, name=None):
        """
        :param lstm_outputs: [batch_size, num_steps, emb_size] 
        :return: [batch_size, num_steps, num_tags]
        """
        with tf.variable_scope("project"  if not name else name):
            
            # project to score of tags
            with tf.variable_scope("logits"):
                W = tf.get_variable("W", shape=[self.cnn_output_width, self.num_tags],
                                    dtype=tf.float32, initializer=self.initializer)

                b = tf.get_variable("b",  initializer=tf.constant(0.001, shape=[self.num_tags]))

                pred = tf.nn.xw_plus_b(idcnn_outputs, W, b)

            return tf.reshape(pred, [-1, self.num_steps, self.num_tags])

    def loss_layer(self, project_logits, lengths, name=None):
        """
        calculate crf loss
        :param project_logits: [1, num_steps, num_tags]
        :return: scalar loss
        """
        with tf.variable_scope("crf_loss"  if not name else name):
            small = -1000.0
            # pad logits for crf loss
            start_logits = tf.concat(
                [small * tf.ones(shape=[self.batch_size, 1, self.num_tags]), tf.zeros(shape=[self.batch_size, 1, 1])], axis=-1)
            pad_logits = tf.cast(small * tf.ones([self.batch_size, self.num_steps, 1]), tf.float32)
            logits = tf.concat([project_logits, pad_logits], axis=-1)
            logits = tf.concat([start_logits, logits], axis=1)
            targets = tf.concat(
                [tf.cast(self.num_tags*tf.ones([self.batch_size, 1]), tf.int32), self.targets], axis=-1)

            self.trans = tf.get_variable(
                "transitions",
                shape=[self.num_tags + 1, self.num_tags + 1],
                initializer=self.initializer)
            log_likelihood, self.trans = crf_log_likelihood(
                inputs=logits,
                tag_indices=targets,
                transition_params=self.trans,
                sequence_lengths=lengths+1)

            return tf.reduce_mean(-log_likelihood)
    def lstm_decode_layer(self, lstm_outputs,name=None):
        """
        calculate lstm decode loss
        :param project_logits: 
        :param lengths: 
        :param name: 
        :return: loss
        """
        with tf.variable_scope("lstm_decode" if not name else name):
            decode_lstm = rnn.LSTMCell(self.lstm_dim,use_peepholes=True)
            init_state = decode_lstm.zero_state(self.batch_size, dtype=tf.float32)
            #tf.constant会报错
            # outputs = [tf.zeros(shape=[self.batch_size,self.lstm_dim])]
            # state = init_state

            i = tf.constant(0)
            while_condition = lambda i,state,outputs: i<self.num_steps
            # self.outputs = tf.Variable(shape=[self.num_steps,self.batch_size, self.lstm_dim])
            self.outputs = tf.zeros([self.num_steps,self.batch_size, self.lstm_dim])
            def body(i,state,outputs):
                # state = init_state
                # if i > 0:
                # tf.get_variable_scope().reuse_variables()
                # 这里的state保存了每一层 LSTM 的状态
                (cell_output, state) = decode_lstm(tf.concat([lstm_outputs[:, i, :], outputs[-1]], axis=-1),state)
                # (cell_output, state) = decode_lstm(lstm_outputs[:, i, :],state)
                with tf.variable_scope("hidden"):
                    W = tf.get_variable("W", shape=[self.lstm_dim, self.lstm_dim],
                                        dtype=tf.float32, initializer=self.initializer)

                    b = tf.get_variable("b", shape=[self.lstm_dim], dtype=tf.float32,
                                        initializer=tf.zeros_initializer())
                    output = tf.reshape(cell_output, shape=[self.batch_size, self.lstm_dim])
                    hidden = tf.tanh(tf.nn.xw_plus_b(output, W, b))
                outputs = tf.concat([outputs[:i],tf.reshape(hidden,shape=[1,-1,self.lstm_dim]),outputs[i+1:]],axis=0)
                # outputs
                return [i+1,state,outputs]

            # do the loop:
            final = tf.while_loop(while_condition, body, [i,init_state,self.outputs])
            self.outputs = final[2]
            outputs = tf.transpose(self.outputs,[1,0,2])
        with tf.variable_scope("logits"):
            W = tf.get_variable("W", shape=[self.lstm_dim, self.num_tags],
                                dtype=tf.float32, initializer=self.initializer)

            b = tf.get_variable("b", shape=[self.num_tags], dtype=tf.float32,
                                initializer=tf.zeros_initializer())
            outputs = tf.reshape(outputs, shape=[-1, self.lstm_dim])
            pred = tf.nn.xw_plus_b(outputs, W, b)

        return tf.reshape(pred, [-1, self.num_steps, self.num_tags])

    def lstm_decode_loss_layer(self, lstm_outputs,bias,name=None):
        # return tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.targets,logits=lstm_outputs))
        with tf.variable_scope("lstm_decode_loss" if not name else name):
            lstm_outputs = tf.nn.softmax(lstm_outputs)
            zero = tf.zeros(shape=[self.batch_size,self.num_steps],dtype=tf.int32)
            one = tf.ones(shape=[self.batch_size,self.num_steps],dtype=tf.float32)
            logpy = tf.reduce_sum(tf.log(lstm_outputs)*tf.cast(tf.one_hot(self.targets,self.num_tags,1,0),tf.float32),axis=-1)
            I0 = tf.cast(tf.equal(zero,self.targets),tf.float32)
            # print(I0.shape)
            I1 = one-I0
            if bias:
                print(-tf.reduce_mean(logpy*I0+bias*logpy*I1))
                return -tf.reduce_mean(logpy*I0+bias*logpy*I1)
            else:
                return -tf.reduce_mean(logpy)


    def create_feed_dict(self, type, batch):
        """
        :param is_train: Flag, True for train batch
        :param batch: list train/evaluate data 
        :return: structured data to feed
        """
        _, chars, segs, tags = batch
        feed_dict = {
            self.char_inputs: np.asarray(chars),
            self.seg_inputs: np.asarray(segs),
            self.dropout: 1.0,
        }
        if type=='train':
            feed_dict[self.targets] = np.asarray(tags)
            feed_dict[self.dropout] = self.config["dropout_keep"]
        elif type=='dev':
            feed_dict[self.targets] = np.asarray(tags)
        return feed_dict

    def run_step(self, sess, type, batch):
        """
        :param sess: session to run the batch
        :param is_train: a flag indicate if it is a train batch
        :param batch: a dict containing batch data
        :return: batch result, loss of the batch or logits
        """
        feed_dict = self.create_feed_dict(type, batch)
        if type=='train':

            global_step, loss, _, summary = sess.run(
                [self.global_step, self.loss, self.train_op, self.summary],
                feed_dict)
            return global_step, loss, summary
        elif type=='dev':
            lengths, logits, loss = sess.run([self.lengths, self.logits,self.loss], feed_dict)
            return lengths, logits, loss
        else:
            lengths, logits = sess.run([self.lengths, self.logits], feed_dict)
            return lengths, logits


    def decode(self, logits, lengths, matrix):
        """
        :param logits: [batch_size, num_steps, num_tags]float32, logits
        :param lengths: [batch_size]int32, real length of each sequence
        :param matrix: transaction matrix for inference
        :return:
        """
        # inference final labels usa viterbi Algorithm
        paths = []
        small = -1000.0
        start = np.asarray([[small]*self.num_tags +[0]])
        for score, length in zip(logits, lengths):
            score = score[:length]
            pad = small * np.ones([length, 1])
            logits = np.concatenate([score, pad], axis=1)
            logits = np.concatenate([start, logits], axis=0)
            path, _ = viterbi_decode(logits, matrix)

            paths.append(path[1:])
        return paths

    def evaluate(self, sess, data_manager, id_to_tag, type):
        """
        :param sess: session  to run the model 
        :param data: list of data
        :param id_to_tag: index to tag name
        :return: evaluate result
        """
        results = []
        if self.decode_method=='crf':
            trans = self.trans.eval()
        losses = []
        for batch in data_manager.iter_batch():
            strings = batch[0]
            tags = batch[-1]
            if type=='test':
                lengths, logits = self.run_step(sess, type, batch)
            else:
                lengths, logits, loss = self.run_step(sess, type, batch)
                losses.append(loss)
            if self.decode_method == 'crf':
                batch_paths = self.decode(logits, lengths, trans)
            else:
                batch_paths = np.argmax(logits, -1)
                # print(batch_paths[0])
            for i in range(len(strings)):
                result = []
                string = strings[i][:lengths[i]]
                gold = DataUtils.iobes_iob([id_to_tag[int(x)] for x in tags[i][:lengths[i]]])
                pred = DataUtils.iobes_iob([id_to_tag[int(x)] for x in batch_paths[i][:lengths[i]]])
                for char, gold, pred in zip(string, gold, pred):
                    result.append(" ".join([char, gold, pred]))
                results.append(result)
        return results,losses

    def evaluate_line(self, sess, inputs, id_to_tag):

        lengths, logits = self.run_step(sess, 'test', inputs)
        if self.decode_method == 'crf':
            trans = self.trans.eval(session=sess)
            batch_paths = self.decode(logits, lengths, trans)
        else:
            batch_paths = np.argmax(logits, -1)
        tags = [id_to_tag[idx] for idx in batch_paths[0]]
        return tags

