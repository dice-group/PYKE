import datetime
import logging
import os
import pickle

import numpy as np

import bz2

triple = 3


def get_path_knowledge_graphs(path: str):
    """

    :param path: str represents path of a KB or path of folder containg KBs
    :return:
    """
    KGs = list()

    if os.path.isfile(path):
        KGs.append(path)
    else:
        for root, dir, files in os.walk(path):
            for file in files:
                print(file)
                if '.nq' in file or '.nt' in file or 'ttl' in file:
                    KGs.append(path + '/' + file)
    if len(KGs) == 0:
        print(path + ' is not a path for a file or a folder containing any .nq or .nt formatted files')
        exit(1)
    return KGs


def file_type(f_name):
    if f_name[-4:] == '.bz2':
        reader = bz2.open(f_name, "rt")
        return reader
    return open(f_name, "r")


def create_experiment_folder():
    directory = os.getcwd() + '/Experiments/'
    folder_name = str(datetime.datetime.now())
    path_of_folder = directory + folder_name
    os.makedirs(path_of_folder)
    return path_of_folder, path_of_folder[:path_of_folder.rfind('/')]


def serializer(*, object_: object, path: str, serialized_name: str):
    with open(path + '/' + serialized_name + ".p", "wb") as f:
        pickle.dump(object_, f)
    f.close()


def deserializer(*, path: str, serialized_name: str):
    with open(path + "/" + serialized_name + ".p", "rb") as f:
        obj_ = pickle.load(f)
    f.close()
    return obj_


def randomly_initialize_embedding_space(num_vocab, embeddings_dim):
    return np.random.rand(num_vocab, embeddings_dim)


def generator_of_reader(bound, knowledge_graphs, rdf_decomposer, ):
    for f_name in knowledge_graphs:
        reader = file_type(f_name)
        total_sentence = 0
        for sentence in reader:
            # Ignore Literals
            if '"' in sentence or "'" in sentence or '# started' in sentence:
                continue
            if len(sentence) < 3:
                continue

            if total_sentence == bound: break
            total_sentence += 1

            try:
                s, p, o, flag = rdf_decomposer(sentence)

                # <..> <..> <..>
                if flag != triple:
                    print(sentence, '+', flag)
                    continue

            except ValueError:
                print('****{0}****'.format(sentence))
                print('value error')
                exit(1)

            yield s, p, o

        reader.close()

def create_logger(*, name, p):
    logger = logging.getLogger(name)

    logger.setLevel(logging.INFO)
    # create file handler which logs even debug messages
    fh = logging.FileHandler(p + '/info.log')
    fh.setLevel(logging.INFO)

    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    # add the handlers to logger
    logger.addHandler(ch)
    logger.addHandler(fh)

    return logger
