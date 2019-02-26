import os
import json
import shutil
import logging

import tensorflow as tf
from conlleval import return_report

models_path = "./models"
eval_path = "./evaluation"
eval_temp = os.path.join(eval_path, "temp")
eval_script = os.path.join(eval_path, "conlleval")


def get_logger(log_file):
    logger = logging.getLogger(log_file)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


# def test_ner(results, path):
#     """
#     Run perl script to evaluate model
#     """
#     script_file = "conlleval"
#     output_file = os.path.join(path, "ner_predict.utf8")
#     result_file = os.path.join(path, "ner_result.utf8")
#     with open(output_file, "w") as f:
#         to_write = []
#         for block in results:
#             for line in block:
#                 to_write.append(line + "\n")
#             to_write.append("\n")
#
#         f.writelines(to_write)
#     os.system("perl {} < {} > {}".format(script_file, output_file, result_file))
#     eval_lines = []
#     with open(result_file) as f:
#         for line in f:
#             eval_lines.append(line.strip())
#     return eval_lines


def test_ner(results, path):
    """
    Run perl script to evaluate model
    """
    output_file = path+"_predict.utf8"
    with open(output_file, "w",encoding='utf8') as f:
        to_write = []
        for block in results:
            for line in block:
                to_write.append(line + "\n")
            to_write.append("\n")

        f.writelines(to_write)
    eval_lines = return_report(output_file)
    return eval_lines



def make_path(params):
    """
    Make folders for training and evaluation
    """
    if not os.path.isdir("../maps"):
        os.makedirs("../maps")
    if not os.path.isdir("../log"):
        os.makedirs("../log")
    if not os.path.isdir("../config"):
        os.makedirs("../config")
    if not os.path.isdir(params.ckpt_path[:-5]):
        os.makedirs(params.ckpt_path[:-5])


def clean(params):
    """
    Clean current folder
    remove saved model and training log 
    """
    if os.path.isfile(params.vocab_file):
        os.remove(params.vocab_file)

    if os.path.isfile(params.map_file):
        os.remove(params.map_file)

    if os.path.isdir(params.ckpt_path[:-6]):
        shutil.rmtree(params.ckpt_path[:-6])

    if os.path.isdir(params.summary_path):
        shutil.rmtree(params.summary_path)

    if os.path.isdir(params.result_path):
        shutil.rmtree(params.result_path)

    if os.path.isdir(params.log_file):
        shutil.rmtree(params.log_file)

    if os.path.isdir("__pycache__"):
        shutil.rmtree("__pycache__")

    if os.path.isfile(params.config_file):
        os.remove(params.config_file)

    if os.path.isfile(params.vocab_file):
        os.remove(params.vocab_file)




def convert_to_text(line):
    """
    Convert conll data to text
    """
    to_print = []
    for item in line:

        try:
            if item[0] == " ":
                to_print.append(" ")
                continue
            word, gold, tag = item.split(" ")
            if tag[0] in "SB":
                to_print.append("[")
            to_print.append(word)
            if tag[0] in "SE":
                to_print.append("@" + tag.split("-")[-1])
                to_print.append("]")
        except:
            print(list(item))
    return "".join(to_print)


def extract(sentence,tags):
    entities = []
    entity = ""
    chunk_start = False
    chunk_end = False
    for i in range(len(tags)):
        if tags[i][0] == "S":
            entities.append({"value": sentence[i], "start": i, "end": i+1, "type":tags[i][2:]})
        if i==0 and tags[i][0]!='O': chunk_start = True
        if tags[i][0] == 'B':chunk_start = True
        if i>0:
            if tags[i-1] == 'O' and tags[i][0] == 'I': chunk_start = True
            if tags[i - 1][0] == 'B' and tags[i][0] == 'B': chunk_end = True
            if tags[i - 1][0] == 'B' and tags[i][0] == 'O': chunk_end = True
            if tags[i - 1][0] == 'I' and tags[i][0] == 'B': chunk_end = True
            if tags[i - 1][0] == 'I' and tags[i][0] == 'O': chunk_end = True
            if tags[i - 1][0] == 'I' and tags[i][0] == 'S': chunk_end = True
            if tags[i - 1][0] == 'E' and tags[i][0] == 'O': chunk_end = True
            if tags[i - 1][0] == 'E' and tags[i][0] == 'B': chunk_end = True
            if tags[i - 1][0] == 'E' and tags[i][0] == 'I': chunk_end = True
            if tags[i - 1][0] == 'E' and tags[i][0] == 'S': chunk_end = True

        if chunk_end or chunk_start:

            if chunk_end:
                entities[-1]['value'] = entity
                entities[-1]['end'] = i
                chunk_end = False
                entity = ""
            if chunk_start:
                entities.append({'type': tags[i][2:], 'start': i})
                entity = sentence[i]
                chunk_start = False

        elif entity:
            entity+=sentence[i]

        if entity and i+1==len(tags):
            # entity+=sentence[i]
            entities[-1]['value'] = entity
            entities[-1]['end'] = i+1
            chunk_end = False
            entity = ""
    return entities

if __name__=='__main__':
    print(extract('武田信廉（1532年~1582年），武田信虎3男，母亲为正室大井夫人。',['B-父母-2', 'I-父母-2', 'I-父母-2', 'I-父母-2', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-父母-2', 'I-父母-2', 'I-父母-2', 'I-父母-2', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']
))
