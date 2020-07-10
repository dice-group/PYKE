import argparse

from helper_classes import *
import numpy as np
import random
import pandas as pd

random_state = 1
np.random.seed(random_state)
random.seed(random_state)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--kg_path", type=str, default="KGs/father.nt", nargs="?",
                        help="Path of dataset.")
    parser.add_argument("--embedding_dim", type=int, default=20, nargs="?",
                        help="Number of dimensions in embedding space.")
    parser.add_argument("--num_iterations", type=int, default=1000, nargs="?",
                        help="Number of iterations.")
    parser.add_argument("--K", type=int, default=100, nargs="?",
                        help="Number of interactions.")
    parser.add_argument("--omega", type=float, default=0.45557, nargs="?",
                        help="Omega: a constant in repulsive force calculation.")
    parser.add_argument("--energy_release", type=float, default=0.0414, nargs="?",
                        help="Energy release per iteration.")

    parser.add_argument("--eval", type=bool, default=True, nargs="?",
                        help="Perform Type prediction.")

    args = parser.parse_args()
    kg_path = args.kg_path

    # DEFINE MODEL PARAMS
    K = args.K
    num_of_dims = args.embedding_dim
    bound_on_iter = args.num_iterations
    omega = args.omega
    e_release = args.energy_release

    flag_for_type_prediction = args.eval

    storage_path, experiment_folder = create_experiment_folder()
    logger = create_logger(name='PYKE', p=storage_path)

    logger.info('Starts')
    logger.info('Hyperparameters:  {0}'.format(args))

    parser = Parser(p_folder=storage_path, k=K)

    parser.set_logger(logger)

    parser.set_similarity_measure(PPMI)

    model = PYKE(logger=logger)

    analyser = DataAnalyser(p_folder=storage_path, logger=logger)

    holder = parser.pipeline_of_preprocessing(kg_path)

    vocab_size = len(holder)

    embeddings = randomly_initialize_embedding_space(vocab_size, num_of_dims)

    learned_embeddings = model.pipeline_of_learning_embeddings(e=embeddings,
                                                               max_iteration=bound_on_iter,
                                                               energy_release_at_epoch=e_release,
                                                               holder=holder, omega=omega)
    del embeddings
    del holder

    vocab = deserializer(path=storage_path, serialized_name='vocabulary')
    learned_embeddings.index = [i for i in vocab]
    learned_embeddings.to_csv(storage_path + '/PYKE_50_embd.csv')

    # This crude workaround performed to serialize dataframe with corresponding terms.
    learned_embeddings.index = [i for i in range(len(vocab))]

    if flag_for_type_prediction:
        analyser.perform_type_prediction(learned_embeddings)
        analyser.perform_clustering_quality(learned_embeddings)

    # analyser.plot2D(learned_embeddings)
