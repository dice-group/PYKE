import os
import re
from collections import Counter, defaultdict
import itertools
from scipy.spatial.distance import cosine
from sklearn.neighbors import NearestNeighbors
import os.path
from numpy import linalg as LA
import numpy as np
import pandas as pd
import time
import warnings
import sys
from abc import ABC, abstractmethod
import hdbscan
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

warnings.filterwarnings('ignore')


def performance_debugger(func_name):
    def function_name_decoratir(func):
        def debug(*args, **kwargs):
            long_string = ''
            starT = time.time()
            print('\n\n######', func_name, ' starts ######')
            r = func(*args, **kwargs)
            print(func_name, ' took ', time.time() - starT, ' seconds\n')
            long_string += str(func_name) + ' took:' + str(time.time() - starT) + ' seconds'

            return r

        return debug

    return function_name_decoratir


class SimilarityCalculator(ABC):
    def __init__(self):
        self._inverted_index = None
        self._num_triples = None

    @abstractmethod
    def get_similarities(self, inverted_index, num_triples, top_K):
        pass


class PPMI(SimilarityCalculator):
    def __init__(self):
        """

        :param co_occurrences: term to list of terms
        :param num_triples:
        """
        super().__init__()
        self._marginal_probs = None

    def calculate_marginal_probabilities(self):
        marginal_probs = dict()
        for unq_ent, list_of_context_ent in enumerate(self.inverted_index):
            # N is multipled by 2 as list_of_context_ent contains other two element of an RDF triple
            probability = len(list_of_context_ent) / (self._num_triples * 2)

            marginal_probs[unq_ent] = probability
        self._marginal_probs = marginal_probs

    @performance_debugger('Calculation of PPMIs')
    def calculate_ppmi(self) -> np.array:

        holder = list()

        for unq_ent, list_of_context_ent in enumerate(self.inverted_index):
            top_k_sim = dict()

            marginal_prob_of_target = self._marginal_probs[unq_ent]

            statistical_info_of_cooccurrences = Counter(list_of_context_ent)

            top_k_sim.setdefault(unq_ent, dict())

            for context_ent, co_occuring_freq in statistical_info_of_cooccurrences.items():

                joint_prob = co_occuring_freq / self._num_triples

                marginal_prob_of_context = self._marginal_probs[context_ent]

                denominator = marginal_prob_of_target * marginal_prob_of_context

                PMI_val = np.log2(joint_prob) - np.log2(denominator)

                if PMI_val <= 0:
                    continue

                if len(top_k_sim[unq_ent]) <= self._topK:
                    top_k_sim[unq_ent][context_ent] = PMI_val.astype(np.float32)
                else:
                    for k, v in top_k_sim[unq_ent].items():
                        if v < PMI_val:
                            top_k_sim[unq_ent][context_ent] = PMI_val
                            del top_k_sim[unq_ent][k]
                            break

            context = np.array(list(top_k_sim[unq_ent].keys()), dtype=np.uint32)
            sims = np.array(list(top_k_sim[unq_ent].values()), dtype=np.float32)
            sims.shape = (sims.size, 1)

            # sampled may contain dublicated variables
            sampled = np.random.choice(len(self.inverted_index), self._topK)

            # negatives must be disjoint from context of k.th vocabulary term and k.term itsel
            negatives = np.setdiff1d(sampled, np.append(context, unq_ent), assume_unique=True)

            holder.append((context, sims, negatives))

        return holder

    def get_similarities(self, inverted_index, num_triples, top_K):
        """

        :param inverted_index:
        :param num_triples:
        :return: similarities data structure is a numpy array of dictionaries.

                i.th element of the numpy array corresponds to i.th element in the vocabulary.
                The dictionary stored in the i.th element:
                    Key: a vocabulary term
                    Val: PPMI value

        """
        self.inverted_index = inverted_index
        self._num_triples = num_triples
        self._topK = top_K
        self.calculate_marginal_probabilities()

        similarities = self.calculate_ppmi()

        return similarities


class Parser:
    def __init__(self, logger=False, p_folder: str = 'not initialized', k=1):
        self.path = 'uninitialized'
        self.logger = logger
        self.p_folder = p_folder
        self.similarity_function = None
        self.similarity_measurer = None
        self.K = int(k)
        self.logger = None

    def set_logger(self, logger):
        self.logger = logger

    def set_similarity_function(self, f):
        self.similarity_function = f

    def set_similarity_measure(self, f):
        self.similarity_measurer = f

    def set_experiment_path(self, p):
        self.p_folder = p

    def set_k_entities(self, k):
        self.K = k

    def get_path_knowledge_graphs(self, path: str):
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
                    if '.nq' in file or '.nt' in file or 'ttl' in file or '.txt' in file:
                        KGs.append(path + '/' + file)
        if len(KGs) == 0:
            self.logger.info(
                '{0} is not a path for a file or a folder containing any .nq or .nt formatted files'.format(path))
            self.logger.info('Execution is terminated.')
            exit(1)
        return KGs

    @staticmethod
    def decompose_rdf(sentence):

        flag = 0

        components = re.findall('<(.+?)>', sentence)
        #components = sentence.split()

        if len(components) == 2:
            s, p = components
            remaining_sentence = sentence[sentence.index(p) + len(p) + 2:]
            literal = remaining_sentence[:-1]
            o = literal
            flag = 2

        elif len(components) == 4:
            del components[-1]
            s, p, o = components

            flag = 4

        elif len(components) == 3:
            s, p, o = components
            flag = 3

        elif len(components) > 4:

            s = components[0]
            p = components[1]
            remaining_sentence = sentence[sentence.index(p) + len(p) + 2:]
            literal = remaining_sentence[:remaining_sentence.index(' <http://')]
            o = literal

        else:
            ## This means that literal contained in RDF triple contains < > symbol
            raise ValueError()

        o = re.sub("\s+", "", o)
        s = re.sub("\s+", "", s)
        p = re.sub("\s+", "", p)

        return s, p, o, flag

    @performance_debugger('Preprocessing')
    def pipeline_of_preprocessing(self, f_name, bound=''):

        inverted_index, num_of_rdf, similar_characteristics = self.inverted_index(f_name, bound)

        holder = self.similarity_measurer().get_similarities(inverted_index, num_of_rdf, self.K)

        return holder

    @performance_debugger('Constructing Inverted Index')
    def inverted_index(self, path, bound):

        inverted_index = {}
        vocabulary = {}
        similar_characteristics = defaultdict(lambda: defaultdict(list))

        num_of_rdf = 0

        type_info = defaultdict(set)

        sentences = generator_of_reader(bound, self.get_path_knowledge_graphs(path), self.decompose_rdf)

        for s, p, o in sentences:

            num_of_rdf += 1

            # mapping from string to vocabulary
            vocabulary.setdefault(s, len(vocabulary))
            vocabulary.setdefault(p, len(vocabulary))
            vocabulary.setdefault(o, len(vocabulary))

            inverted_index.setdefault(vocabulary[s], []).extend([vocabulary[o], vocabulary[p]])
            inverted_index.setdefault(vocabulary[p], []).extend([vocabulary[s], vocabulary[o]])
            inverted_index.setdefault(vocabulary[o], []).extend([vocabulary[s], vocabulary[p]])

            if 'rdf-syntax-ns#type' in p:
                type_info[vocabulary[s]].add(vocabulary[o])

        self.logger.info('Number of RDF triples:\t{0}'.format(num_of_rdf))
        self.logger.info('Number of vocabulary terms:\t{0}'.format(len(vocabulary)))
        self.logger.info('Number of subjects with type information:\t{0}'.format(len(type_info)))
        self.logger.info('Number of types :\t{0}'.format(len(set(itertools.chain.from_iterable(type_info.values())))))

        if num_of_rdf == 0:
            self.logger.info('Exception at parsing dataset: No RDF triple processed.')
            self.logger.info('Terminating')
            exit(1)

        assert list(inverted_index.keys()) == list(range(0, len(vocabulary)))

        vocabulary = list(vocabulary.keys())
        self.logger.info('Vocabulary being serialized. Note that ith vocabulary has ith. representation')
        serializer(object_=vocabulary, path=self.p_folder, serialized_name='vocabulary')
        del vocabulary

        inverted_index = list(inverted_index.values())
        self.logger.info('Inverted Index being serialized. Note that ith vocabulary term has ith. document')
        serializer(object_=inverted_index, path=self.p_folder, serialized_name='inverted_index')

        serializer(object_=type_info, path=self.p_folder, serialized_name='type_info')
        del type_info

        return inverted_index, num_of_rdf, similar_characteristics


class PYKE(object):
    def __init__(self, epsilon=0.01, logger=None):

        self.epsilon = epsilon
        self.ratio = list()
        self.system_energy = 1
        self.logger = logger

    @staticmethod
    def apply_hooke_s_law(embedding_space, target_index, context_indexes, PMS):
        """

        :param embedding_space:
        :param target_index:
        :param context_indexes:
        :param PMS:
        :return:
        """

        dist = embedding_space[context_indexes] - embedding_space[target_index]
        # replace all zeros to 1 as a normalizer.
        dist[dist == 0] = 0.01
        # replace all
        pull = dist * PMS
        total_pull = np.sum(pull, axis=0)

        return total_pull, np.abs(dist).sum()

    @staticmethod
    def apply_inverse_hooke_s_law(embedding_space, target_index, repulsive_indexes, omega):
        """

        :param embedding_space:
        :param target_index:
        :param repulsive_indexes:
        :param negative_constant:
        :return:
        """

        # calculate distance from target to repulsive entities
        dist = embedding_space[repulsive_indexes] - embedding_space[target_index]

        # replace all zeros to 1
        dist[dist == 0] = 0.01

        with warnings.catch_warnings():
            try:

                total_push = -omega * np.reciprocal(dist).sum(axis=0)

                # replace all zeros to 1 if needed
                # total_push[total_push == 0] = 0.01

            except RuntimeWarning as r:
                print(r)
                print("Unexpected error:", sys.exc_info()[0])
                exit(1)

        return total_push, np.abs(dist).sum()

    def go_through_entities(self, e, holder, omega):

        sum_pos_sem_dist = 0
        sum_neg_sem_dist = 0
        for target_index in range(len(e)):
            indexes_of_attractive, pms_of_contest, indexes_of_repulsive = holder[target_index]

            pull, p = self.apply_hooke_s_law(e, target_index, indexes_of_attractive, pms_of_contest)

            sum_pos_sem_dist += p

            push, n = self.apply_inverse_hooke_s_law(e, target_index, indexes_of_repulsive,
                                                     omega)
            sum_neg_sem_dist += n

            total_effect = (pull + push) * self.system_energy

            e[target_index] = e[target_index] + total_effect

        semantic_distance = dict()
        semantic_distance['pos'] = sum_pos_sem_dist
        semantic_distance['neg'] = sum_neg_sem_dist

        return e, semantic_distance

    @performance_debugger('Generating Embeddings:')
    def pipeline_of_learning_embeddings(self, *, e, max_iteration, energy_release_at_epoch, holder, omega):

        for epoch in range(max_iteration):
            self.logger.info('EPOCH: {0}'.format(epoch))

            previous_f_norm = LA.norm(e, 'fro')

            e, semantic_dist = self.go_through_entities(e, holder, omega)

            self.system_energy = self.system_energy - energy_release_at_epoch

            self.logger.info(
                'Distance:{0}\t System Energy:{1} \t Distance Ratio:{2}'.format(semantic_dist, self.system_energy,
                                                                                semantic_dist['pos'] / semantic_dist[
                                                                                    'neg']))

            e = np.nan_to_num(e)

            with warnings.catch_warnings():
                try:
                    e = (e - e.min(axis=0)) / (e.max(axis=0) - e.min(axis=0))
                except RuntimeWarning as r:
                    print(r)
                    print(e.mean())
                    print(np.isnan(e).any())
                    print(np.isinf(e).any())
                    exit(1)

            new_f_norm = LA.norm(e, 'fro')

            if self.equilibrium(epoch, previous_f_norm, new_f_norm):
                e = np.nan_to_num(e)
                break

        return pd.DataFrame(e)

    def equilibrium(self, epoch, p_n, n_n):
        val = np.abs(p_n - n_n)

        # or d_ratio < 0.1
        if val < self.epsilon or self.system_energy <= 0:
            self.logger.info('Equilibrium is reached.\t Epoch: {0}\t System Energy:{1}\t Euclidiean distance between '
                             'last two representations: {2}'.format(epoch,self.system_energy,val))
            return True
        return False


class DataAnalyser(object):
    def __init__(self, p_folder: str = 'not initialized',logger=None):

        self.p_folder = p_folder
        self.kg_path = self.p_folder
        self.logger=logger

    def set_experiment_path(self, p):
        self.p_folder = p

    @staticmethod
    def calculate_euclidean_distance(*, embeddings, entitiy_to_P_URI, entitiy_to_N_URI):
        """
        Calculate the difference
        Target entitiy
        Attractive entitiy entitiy_to_P_URI list of dictionaries
        Repulsive entitiy
        """

        total_distance_from_attractives = 0
        total_distance_from_repulsives = 0

        for index in range(len(entitiy_to_P_URI)):
            index_of_attractive_entitites = np.array(list(entitiy_to_P_URI[index].keys()), dtype=np.int32)
            index_of_repulsive_entitites = np.array(list(entitiy_to_N_URI[index]), dtype=np.int32)

            total_distance_from_attractives += np.linalg.norm(
                embeddings[index_of_attractive_entitites] - embeddings[index])

            total_distance_from_repulsives += np.linalg.norm(
                embeddings[index_of_repulsive_entitites] - embeddings[index])

        print('Distance comparision d(A)/d(R) ', total_distance_from_attractives / total_distance_from_repulsives)

    @performance_debugger('Pseudo labeling via HDBSCAN')
    def pseudo_label_HDBSCAN(self, df, min_cluster_size=None, min_samples=None):
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples).fit(df)
        df['labels'] = clusterer.labels_
        return df

    @performance_debugger('Cluster Quality')
    def perform_clustering_quality(self, df):
        """

        :param df:
        :param type_info:
        :param min_cluster_size:
        :param min_samples:
        :return:
        """

        def create_binary_type_vector(t_types, a_types):
            vector = np.zeros(len(all_types))
            i = [a_types.index(_) for _ in t_types]
            vector[i] = 1
            return vector

        type_info = deserializer(path=self.p_folder, serialized_name='type_info')

        # get all unique types, i.e. all o : (s,#type,o) \in KG
        all_types = sorted(set.union(*list(type_info.values())))

        # get only those resources that have type information
        df_only_subjects = df.iloc[list(type_info.keys())]

        # Apply clustering
        df_only_subjects = self.pseudo_label_HDBSCAN(df_only_subjects, min_cluster_size=26, min_samples=29)

        clusters = pd.unique(df_only_subjects.labels)

        sum_purity = 0
        for c in clusters:

            valid_indexes_in_c = df_only_subjects[df_only_subjects.labels == c].index.values

            sum_of_cosines = 0

            self.logger.info('##### CLUSTER {0} #####'.format(c))

            for i in valid_indexes_in_c:

                # returns a set of indexes
                types_i = type_info[i]

                vector_type_i = create_binary_type_vector(types_i, all_types)

                for j in valid_indexes_in_c:
                    types_j = type_info[j]
                    vector_type_j = create_binary_type_vector(types_j, all_types)

                    sum_of_cosines += 1 - cosine(vector_type_i, vector_type_j)

            purity = sum_of_cosines / (len(valid_indexes_in_c) ** 2)

            sum_purity += purity

        mean_of_scores = sum_purity / len(clusters)
        self.logger.info('Mean of cluster purity:{0}'.format(mean_of_scores))

    @performance_debugger('Type Prediction')
    def perform_type_prediction(self, df, based_on_num_neigh=3):

        def create_binary_type_vector(t_types, a_types):
            vector = np.zeros(len(all_types))
            i = [a_types.index(_) for _ in t_types]
            vector[i] = 1
            return vector

        def create_binary_type_prediction_vector(t_types, a_types):
            vector = np.zeros(len(all_types))
            i = [a_types.index(_) for _ in itertools.chain.from_iterable(t_types)]
            vector[i] += 1
            return vector

        # get the types. Mapping from the index of subject to the index of object
        type_info = deserializer(path=self.p_folder, serialized_name='type_info')

        # get the index of objects / get type information =>>> s #type o
        all_types = sorted(set.union(*list(type_info.values())))

        # Consider only points with type infos.
        e_w_types = df.loc[list(type_info.keys())]

        neigh = NearestNeighbors(n_neighbors=based_on_num_neigh, algorithm='kd_tree', metric='euclidean',
                                 n_jobs=-1).fit(
            e_w_types)

        # Get similarity results for selected entities
        df_most_similars = pd.DataFrame(neigh.kneighbors(e_w_types, return_distance=False))

        # Reindex the target
        df_most_similars.index = e_w_types.index.values

        # As sklearn implementation of kneighbors returns the point itself as most similar point
        df_most_similars.drop(columns=[0], inplace=True)

        # Map back to the original indexes. KNN does not consider the index of Dataframe.
        mapper = dict(zip(list(range(len(e_w_types))), e_w_types.index.values))
        # The values of most similars are mapped to original vocabulary positions
        df_most_similars = df_most_similars.applymap(lambda x: mapper[x])

        k_values = [1, 3, 5, 10, 15, 30, 50, 100]

        self.logger.info('K values: {0}'.format(k_values))
        for k in k_values:
            self.logger.info('##### {0} #####'.format(k))
            similarities = list()
            for _, S in df_most_similars.iterrows():
                true_types = type_info[_]
                type_predictions = [type_info[_] for _ in S.values[:k]]

                vector_true = create_binary_type_vector(true_types, all_types)
                vector_prediction = create_binary_type_prediction_vector(type_predictions, all_types)

                sim = cosine(vector_true, vector_prediction)
                similarities.append(1 - sim)

            report = pd.DataFrame(similarities)
            self.logger.info('Mean type prediction: {0}'.format(report.mean().values))

    def plot2D(self, df):
        pca = PCA(n_components=2)
        low = pca.fit_transform(df.to_numpy())

        x = low[:, 0]
        y = low[:, 1]
        plt.scatter(x, y)

        for i, txt in enumerate(df.index.tolist()):
            plt.annotate(txt, (x[i], y[i]))

        plt.show()

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
