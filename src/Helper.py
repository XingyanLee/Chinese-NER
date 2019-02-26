"""

Helper.py辅助函数

"""
import numpy as np
import re
import math
import codecs
import random
import os
import pickle
import itertools

import jieba
jieba.initialize()
from six.moves import xrange

# ##将char转换为id并保存
# def store_char2id(inputs,outputs):
#     with open(inputs,encoding='utf-8') as r:
#         char_list = []
#         for w in r.read().split('\n')[1:]:
#             char_list.append(w.split(' ')[0])
#         char_list.insert(0,'<PAD>')
#         char_list.append('<UNKNOW>')
#     with open(outputs, 'w', encoding='utf-8') as w:
#         for i,word in enumerate(char_list):
#             w.write(word+' '+str(i)+'\n')

class DataUtils(object):
    @staticmethod
    def create_dico(item_list):
        """
        Create a dictionary of items from a list of list of items.
        """
        assert type(item_list) is list
        dico = {}
        for items in item_list:
            for item in items:
                if item not in dico:
                    dico[item] = 1
                else:
                    dico[item] += 1
        return dico

    @staticmethod
    def create_mapping(dico):
        """
        Create a mapping (item to ID / ID to item) from a dictionary.
        Items are ordered by decreasing frequency.
        """
        sorted_items = sorted(dico.items(), key=lambda x: (-x[1], x[0]))
        id_to_item = {i: v[0] for i, v in enumerate(sorted_items)}
        item_to_id = {v: k for k, v in id_to_item.items()}
        return item_to_id, id_to_item

    @staticmethod
    def zero_digits(s):
        """
        Replace every digit in a string by a zero.
        """
        return re.sub('\d', '0', s)

    @staticmethod
    def iob2(tags):
        """
        Check that tags have a valid IOB format.
        Tags in IOB1 format are converted to IOB2.
        """
        for i, tag in enumerate(tags):
            if tag == 'O':
                continue
            split = tag.split('-')
            if len(split) != 2 or split[0] not in ['I', 'B']:
                return False
            if split[0] == 'B':
                continue
            elif i == 0 or tags[i - 1] == 'O':  # conversion IOB1 to IOB2
                tags[i] = 'B' + tag[1:]
            elif tags[i - 1][1:] == tag[1:]:
                continue
            else:  # conversion IOB1 to IOB2
                tags[i] = 'B' + tag[1:]
        return True

    @staticmethod
    def iob_iobes(tags):
        """
        IOB -> IOBES
        """
        new_tags = []
        for i, tag in enumerate(tags):
            if tag == 'O':
                new_tags.append(tag)
            elif tag.split('-')[0] == 'B':
                if i + 1 != len(tags) and \
                                tags[i + 1].split('-')[0] == 'I':
                    new_tags.append(tag)
                else:
                    new_tags.append(tag.replace('B-', 'S-'))
            elif tag.split('-')[0] == 'I':
                if i + 1 < len(tags) and \
                                tags[i + 1].split('-')[0] == 'I':
                    new_tags.append(tag)
                else:
                    new_tags.append(tag.replace('I-', 'E-'))
            else:
                raise Exception('Invalid IOB format!')
        return new_tags

    @staticmethod
    def iobes_iob(tags):
        """
        IOBES -> IOB
        """
        new_tags = []
        for i, tag in enumerate(tags):
            if tag.split('-')[0] == 'B':
                new_tags.append(tag)
            elif tag.split('-')[0] == 'I':
                new_tags.append(tag)
            elif tag.split('-')[0] == 'S':
                new_tags.append(tag.replace('S-', 'B-'))
            elif tag.split('-')[0] == 'E':
                new_tags.append(tag.replace('E-', 'I-'))
            elif tag.split('-')[0] == 'O':
                new_tags.append(tag)
            else:
                raise Exception('Invalid format!')
        return new_tags

    @staticmethod
    def insert_singletons(words, singletons, p=0.5):
        """
        Replace singletons by the unknown word with a probability p.
        """
        new_words = []
        for word in words:
            if word in singletons and np.random.uniform() < p:
                new_words.append(0)
            else:
                new_words.append(word)
        return new_words

    @staticmethod
    def get_seg_features(string):
        """
        Segment text with jieba
        features are represented in bies format
        s donates single word
        """
        seg_feature = []

        for word in jieba.cut(string):
            if len(word) == 1:
                seg_feature.append(0)
            else:
                tmp = [2] * len(word)
                tmp[0] = 1
                tmp[-1] = 3
                seg_feature.extend(tmp)
        return seg_feature

    @staticmethod
    def create_input(data):
        """
        Take sentence data and return an input for
        the training or the evaluation function.
        """
        inputs = list()
        inputs.append(data['chars'])
        inputs.append(data["segs"])
        inputs.append(data['tags'])
        return inputs

    @staticmethod
    def load_word2vec(emb_path, id_to_word, word_dim, old_weights):
        """
        Load word embedding from pre-trained file
        embedding size must match
        """
        new_weights = old_weights
        print('Loading pretrained embeddings from {}...'.format(emb_path))
        pre_trained = {}
        emb_invalid = 0
        for i, line in enumerate(codecs.open(emb_path, 'r', 'utf-8')):
            line = line.rstrip().split()
            if len(line) == word_dim + 1:
                pre_trained[line[0]] = np.array(
                    [float(x) for x in line[1:]]
                ).astype(np.float32)
            else:
                emb_invalid += 1
        if emb_invalid > 0:
            print('WARNING: %i invalid lines' % emb_invalid)
        c_found = 0
        c_lower = 0
        c_zeros = 0
        n_words = len(id_to_word)
        # Lookup table initialization
        for i in range(n_words):
            word = id_to_word[i]
            if word in pre_trained:
                new_weights[i] = pre_trained[word]
                c_found += 1
            elif word.lower() in pre_trained:
                new_weights[i] = pre_trained[word.lower()]
                c_lower += 1
            elif re.sub('\d', '0', word.lower()) in pre_trained:
                new_weights[i] = pre_trained[
                    re.sub('\d', '0', word.lower())
                ]
                c_zeros += 1
        print('Loaded %i pretrained embeddings.' % len(pre_trained))
        print('%i / %i (%.4f%%) words have been initialized with '
              'pretrained embeddings.' % (
                  c_found + c_lower + c_zeros, n_words,
                  100. * (c_found + c_lower + c_zeros) / n_words)
              )
        print('%i found directly, %i after lowercasing, '
              '%i after lowercasing + zero.' % (
                  c_found, c_lower, c_zeros
              ))
        return new_weights

    @staticmethod
    def full_to_half(s):
        """
        Convert full-width character to half-width one 
        """
        n = []
        for char in s:
            num = ord(char)
            if num == 0x3000:
                num = 32
            elif 0xFF01 <= num <= 0xFF5E:
                num -= 0xfee0
            char = chr(num)
            n.append(char)
        return ''.join(n)

    @staticmethod
    def cut_to_sentence(text):
        """
        Cut text to sentences 
        """
        sentence = []
        sentences = []
        len_p = len(text)
        pre_cut = False
        for idx, word in enumerate(text):
            sentence.append(word)
            cut = False
            if pre_cut:
                cut = True
                pre_cut = False
            if str(word) in u"。;!?\n":
                cut = True
                if len_p > idx + 1:
                    if text[idx + 1] in ".。”\"\'“”‘’?!":
                        cut = False
                        pre_cut = True

            if cut:
                sentences.append(sentence)
                sentence = []
        if sentence:
            sentences.append("".join(list(sentence)))
        return sentences

    @staticmethod
    def replace_html(s):
        s = s.replace('&quot;', '"')
        s = s.replace('&amp;', '&')
        s = s.replace('&lt;', '<')
        s = s.replace('&gt;', '>')
        s = s.replace('&nbsp;', ' ')
        s = s.replace("&ldquo;", "“")
        s = s.replace("&rdquo;", "”")
        s = s.replace("&mdash;", "")
        s = s.replace("\xa0", " ")
        return (s)

    @staticmethod
    def input_from_line(line, char_to_id):
        # print(line)
        """
        Take sentence data and return an input for
        the training or the evaluation function.
        """
        line = DataUtils.full_to_half(line)
        line = DataUtils.replace_html(line)
        inputs = list()
        inputs.append([line])
        line.replace(" ", "$")
        inputs.append([[char_to_id[char] if char in char_to_id else char_to_id["<UNK>"]
                        for char in line]])
        inputs.append([DataUtils.get_seg_features(line)])
        inputs.append([[]])
        return inputs

class LoaderUtils(object):
    @staticmethod
    def load_sentences(path, lower, zeros):
        """
        Load sentences. A line must contain at least a word and its tag.
        Sentences are separated by empty lines.
        """
        sentences = []
        sentence = []
        num = 0
        for line in codecs.open(path, 'r', 'utf8'):
            num += 1
            line = DataUtils.zero_digits(line.rstrip()) if zeros else line.rstrip()
            # print(list(line))
            if not line:
                if len(sentence) > 0:
                    if 'DOCSTART' not in sentence[0][0]:
                        sentences.append(sentence)
                    sentence = []
            else:
                if line[0] == " ":
                    line = "$" + line[1:]
                    word = line.split()
                    # word[0] = " "
                else:
                    word = line.split()
                    if len(word)<2 and len(line)>2:
                        word = [' ',word[0]]
                assert len(word) >= 2, print([word[0]])
                sentence.append([word[0].lower(),word[1]] if lower else word)
        if len(sentence) > 0:
            if 'DOCSTART' not in sentence[0][0]:
                sentences.append(sentence)
        return sentences

    @staticmethod
    def update_tag_scheme(sentences, tag_scheme):
        """
        Check and update sentences tagging scheme to IOB2.
        Only IOB1 and IOB2 schemes are accepted.
        """
        for i, s in enumerate(sentences):
            tags = [w[-1] for w in s]
            # Check that tags are given in the IOB format
            if not DataUtils.iob2(tags):
                s_str = '\n'.join(' '.join(w) for w in s)
                raise Exception('Sentences should be given in IOB format! ' +
                                'Please check sentence %i:\n%s' % (i, s_str))
            if tag_scheme.lower() == 'iob':
                # If format was IOB1, we convert to IOB2
                for word, new_tag in zip(s, tags):
                    word[-1] = new_tag
            elif tag_scheme.lower() == 'iobes':
                new_tags = DataUtils.iob_iobes(tags)
                for word, new_tag in zip(s, new_tags):
                    word[-1] = new_tag
            else:
                raise Exception('Unknown tagging scheme!')

    @staticmethod
    def char_mapping(sentences, lower):
        """
        Create a dictionary and a mapping of words, sorted by frequency.
        """
        chars = [[x[0].lower() if lower else x[0] for x in s] for s in sentences]
        dico = DataUtils.create_dico(chars)
        dico["<PAD>"] = 10000001
        dico['<UNK>'] = 10000000
        char_to_id, id_to_char = DataUtils.create_mapping(dico)
        print("Found %i unique words (%i in total)" % (
            len(dico), sum(len(x) for x in chars)
        ))
        return dico, char_to_id, id_to_char

    @staticmethod
    def tag_mapping(sentences):
        """
        Create a dictionary and a mapping of tags, sorted by frequency.
        """
        tags = [[char[-1] for char in s] for s in sentences]
        dico = DataUtils.create_dico(tags)
        tag_to_id, id_to_tag = DataUtils.create_mapping(dico)
        print("Found %i unique named entity tags" % len(dico))
        return dico, tag_to_id, id_to_tag

    @staticmethod
    def prepare_dataset(sentences, char_to_id, tag_to_id, lower=False):
        """
        Prepare the dataset. Return a list of lists of dictionaries containing:
            - word indexes
            - word char indexes
            - tag indexes
        """


        def f(x):
            return x.lower() if lower else x

        data = []
        for s in sentences:
            string = [w[0] for w in s]
            chars = [char_to_id[f(w) if f(w) in char_to_id else '<UNK>']
                     for w in string]
            segs = DataUtils.get_seg_features("".join(string))

            tags = [tag_to_id[w[-1]] for w in s]
            # else:
            #     tags = [none_index for _ in chars]
            data.append([string, chars, segs, tags])

        return data

    @staticmethod
    def augment_with_pretrained(dictionary, ext_emb_path, chars):
        """
        Augment the dictionary with words that have a pretrained embedding.
        If `words` is None, we add every word that has a pretrained embedding
        to the dictionary, otherwise, we only add the words that are given by
        `words` (typically the words in the development and test sets.)
        """
        print('Loading pretrained embeddings from %s...' % ext_emb_path)
        assert os.path.isfile(ext_emb_path)

        # Load pretrained embeddings from file
        pretrained = set([
            line.rstrip().split()[0].strip()
            for line in codecs.open(ext_emb_path, 'r', 'utf-8')
            if len(ext_emb_path) > 0
        ])

        # We either add every word in the pretrained file,
        # or only words given in the `words` list to which
        # we can assign a pretrained embedding
        if chars is None:
            for char in pretrained:
                if char not in dictionary:
                    dictionary[char] = 0
        else:
            for char in chars:
                if any(x in pretrained for x in [
                    char,
                    char.lower(),
                    re.sub('\d', '0', char.lower())
                ]) and char not in dictionary:
                    dictionary[char] = 0

        word_to_id, id_to_word = DataUtils.create_mapping(dictionary)
        return dictionary, word_to_id, id_to_word

    @staticmethod
    def save_maps(save_path, *params):
        """
        Save mappings and invert mappings
        """
        pass
        # with codecs.open(save_path, "w", encoding="utf8") as f:
        #     pickle.dump(params, f)

    @staticmethod
    def load_maps(save_path):
        """
        Load mappings from the file
        """
        pass
        # with codecs.open(save_path, "r", encoding="utf8") as f:
        #     pickle.load(save_path, f)

class BatchManager(object):

    def __init__(self, data,  batch_size):
        self.batch_data = self.sort_and_pad(data, batch_size)
        self.len_data = len(self.batch_data)

    def sort_and_pad(self, data, batch_size):
        num_batch = int(math.ceil(len(data) /batch_size))
        sorted_data = sorted(data, key=lambda x: len(x[0]))
        batch_data = list()
        for i in range(num_batch):
            batch_data.append(self.pad_data(sorted_data[i*batch_size : (i+1)*batch_size]))
        return batch_data

    @staticmethod
    def pad_data(data):
        strings = []
        chars = []
        segs = []
        targets = []
        max_length = max([len(sentence[0]) for sentence in data])
        for line in data:
            string, char, seg, target = line
            padding = [0] * (max_length - len(string))
            strings.append(string + padding)
            chars.append(char + padding)
            segs.append(seg + padding)
            targets.append(target + padding)
        return [strings, chars, segs, targets]

    def iter_batch(self, shuffle=False):
        if shuffle:
            random.shuffle(self.batch_data)
        for idx in range(self.len_data):
            yield self.batch_data[idx]

class Helper(object):

    def load_train(self, train_file, lower, zeros):
        self.train_sentences = LoaderUtils.load_sentences(train_file, lower, zeros)

    def load_dev(self, dev_file, lower, zeros):
        self.dev_sentences = LoaderUtils.load_sentences(dev_file, lower, zeros)

    def load_test(self, test_file, lower, zeros):
        self.test_sentences = LoaderUtils.load_sentences(test_file, lower, zeros)

    def select_tag_schema(self,tag_schema='iob'):
        """
        tags classes can switch (IOB / IOBES)
        :param tags: 
        :return: 
        """
        LoaderUtils.update_tag_scheme(self.train_sentences, tag_schema)
        LoaderUtils.update_tag_scheme(self.dev_sentences, tag_schema)
        LoaderUtils.update_tag_scheme(self.test_sentences, tag_schema)

    def load_pretrained_emb(self,emb_file):
        self.emb_file = emb_file

    def create_maps(self,map_file=None,lower=False):
        # create maps if not exist
        if not os.path.isfile(map_file):
            # create dictionary for word
            if self.emb_file:
                dico_chars_train = LoaderUtils.char_mapping(self.train_sentences, lower)[0]
                dico_chars, self.char_to_id, self.id_to_char = LoaderUtils.augment_with_pretrained(
                    dico_chars_train.copy(),
                    self.emb_file,
                    list(itertools.chain.from_iterable(
                        [[w[0] for w in s] for s in self.test_sentences])
                    )
                )
            else:
                _c, self.char_to_id, self.id_to_char = LoaderUtils.char_mapping(self.train_sentences, lower)

            # Create a dictionary and a mapping for tags
            _t, self.tag_to_id, self.id_to_tag = LoaderUtils.tag_mapping(self.train_sentences)
            with open(map_file, "wb") as f:
                pickle.dump([self.char_to_id, self.id_to_char, self.tag_to_id, self.id_to_tag], f)
        else:
            with open(map_file, "rb") as f:
                self.char_to_id, self.id_to_char, self.tag_to_id, self.id_to_tag = pickle.load(f)
        # prepare data, get a collection of list containing index
        self.train_data = LoaderUtils.prepare_dataset(
            self.train_sentences, self.char_to_id, self.tag_to_id, lower
        )
        self.dev_data = LoaderUtils.prepare_dataset(
            self.dev_sentences, self.char_to_id, self.tag_to_id, lower
        )
        self.test_data = LoaderUtils.prepare_dataset(
            self.test_sentences, self.char_to_id, self.tag_to_id, lower
        )
        print("%i / %i / %i sentences in train / dev / test." % (
            len(self.train_data), len(self.dev_data), len(self.test_data)))

    def create_batch(self,batch_size):
        self.train_batch = BatchManager(self.train_data, batch_size)
        self.dev_batch = BatchManager(self.dev_data, 100)
        self.test_batch = BatchManager(self.test_data, 100)


# if __name__=='__main__':
#     with open('/home/shaohui/project/label_sequence/origin_data/jiaotong', "r") as r:
#         sentences = []
#         for line in r:
#             sentences.extend(re.split('(?:。|？)',line.rstrip()))
#         print(len(sentences))
#         sentences1 = []
#         for sentence in sentences:
#             if len(sentence)>10:
#                 sentences1.append(sentence)
#     with open('../origin_data/jiaotong.utf8','w',encoding='utf8') as w:
#         for sentence in sentences1:
#             w.write(sentence+'\n')