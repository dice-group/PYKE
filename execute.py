from helper_classes import PYKE
from helper_classes import Parser
from helper_classes import DataAnalyser
from helper_classes import PPMI

import util as ut
import numpy as np
import random

random_state = 1
np.random.seed(random_state)
random.seed(random_state)

# DEFINE MODEL PARAMS
K = 45
num_of_dims = 50
bound_on_iter = 30
omega = 0.45557
e_release = 0.0414

kg_root = 'KGs/data'
kg_path = kg_root + '/father_someonly.nt'
# For N-Quads, please set ut.triple=4. By default ut.triple=3 as KG is N3.
#ut.triple = 3

storage_path, experiment_folder = ut.create_experiment_folder()

parser = Parser(p_folder=storage_path, k=K)

parser.set_similarity_measure(PPMI)

model = PYKE()

analyser = DataAnalyser(p_folder=storage_path)

holder = parser.pipeline_of_preprocessing(kg_path,bound=1000)

vocab_size = len(holder)

embeddings = ut.randomly_initialize_embedding_space(vocab_size, num_of_dims)

learned_embeddings = model.pipeline_of_learning_embeddings(e=embeddings,
                                                           max_iteration=bound_on_iter,
                                                           energy_release_at_epoch=e_release,
                                                           holder=holder, omega=omega)
del embeddings
del holder

analyser.perform_clustering_quality(learned_embeddings)
analyser.perform_type_prediction(learned_embeddings)

vocab = ut.deserializer(path=storage_path, serialized_name='vocabulary')
learned_embeddings.index=vocab
learned_embeddings.to_csv(storage_path + '/PYKE_50_embd.csv')