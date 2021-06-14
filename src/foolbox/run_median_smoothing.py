import os
os.environ['MKL_THREADING_LAYER'] = 'GNU'
import logging
import torch
import numpy as np
import hashlib
from sacred import Experiment
import seml

from src.datasets.Dataset import get_dataset
from src.models.model_loader import load_model
from src.foolbox.test_attack_median_smoothing import test_attack_median_smoothing

ex = Experiment()
seml.setup_logger(ex)


@ex.post_run_hook
def collect_stats(_run):
    seml.collect_exp_stats(_run)


@ex.config
def config():
    overwrite = None
    db_collection = None
    if db_collection is not None:
        ex.observers.append(seml.create_mongodb_observer(db_collection, overwrite=overwrite))


@ex.automain
def run(
        # Dataset parameters
        id_dataset_name,  # in-dataset name.  string
        ood_dataset_name,  # out-dataset name.  string

        # Model parameters
        directory_model,  # Directory of the model. string
        name_model,  # Model name. string
        model_type,  # Model type. string

        # Attack parameters
        directory_results,  # Directory to store attacks . string
        attack_name,  # Attack name. string
        magnitude,  # Magnitude of attacks. float
        attack_loss,  # Name of attacked loss. string
        punished_measure,  # Name of punished measure. string
        start_data,  # Name of the start data. string. options: in, out or random
        bounds,  # Bounds for the model input. list of 2 int

        adv_train_eps=None,
        n_attacked_data=None): # Number of attacked data in the test set.
    if adv_train_eps is not None:
        name_model = name_model.replace('ADV_TRAIN_EPS', str(adv_train_eps))

    logging.info('Received the following configuration:')
    logging.info(f'DATASET | '
                 f'dataset_name {id_dataset_name} - '
                 f'ood_dataset_name {ood_dataset_name}')
    logging.info(f'ARCHITECTURE | '
                 f' directory_model {directory_model} - '
                 f' name_model {name_model} - '
                 f' model_type {model_type}')
    logging.info(f'TRAINING | '
                 f' attack_name {attack_name} - '
                 f' magnitude {magnitude} - '
                 f' attack_loss {attack_loss} - '
                 f' punished_measure {punished_measure} - '
                 f' start_data {start_data} - '
                 f' bounds {bounds}')
    config_dict = torch.load(f'{directory_model}{name_model}')['model_config_dict']
    
    ###################
    ## Load datasets ##
    ###################
    if attack_name == 'Gaussian_L2_Noise':
        id_train_loader, id_val_loader, id_test_loader, N = get_dataset(id_dataset_name,
                                                                        batch_size=1,
                                                                        batch_size_eval=1,
                                                                        split=config_dict['split'],
                                                                        seed=config_dict['seed_dataset'],
                                                                        n_test_data=n_attacked_data)
        ood_train_loader, ood_val_loader, ood_test_loader, N = get_dataset(ood_dataset_name,
                                                                           batch_size=1,
                                                                           batch_size_eval=1,
                                                                           split=config_dict['split'],
                                                                           seed=config_dict['seed_dataset'],
                                                                           n_test_data=n_attacked_data)

    else:
        id_train_loader, id_val_loader, id_test_loader, N = get_dataset(id_dataset_name,
                                                                        batch_size=512,
                                                                        batch_size_eval=1,
                                                                        split=config_dict['split'],
                                                                        seed=config_dict['seed_dataset'],
                                                                        n_test_data=n_attacked_data)
        ood_train_loader, ood_val_loader, ood_test_loader, N = get_dataset(ood_dataset_name,
                                                                           batch_size=512,
                                                                           batch_size_eval=1,
                                                                           split=config_dict['split'],
                                                                           seed=config_dict['seed_dataset'],
                                                                           n_test_data=n_attacked_data)

    #################
    ## Load model ##
    #################
    model = load_model(directory_model=directory_model,
                       name_model=name_model,
                       model_type=model_type)

    ##################
    ## Attack model ##
    ##################
    full_config_dict = {'dataset_name': id_dataset_name,
                        'ood_dataset_name': ood_dataset_name,
                        'name_model': name_model,
                        'attack_name': attack_name,
                        'magnitude': magnitude,
                        'attack_loss': attack_loss,
                        'punished_measure': punished_measure,
                        'start_data': start_data}
    full_config_name = ''
    for k, v in full_config_dict.items():
        full_config_name += str(v) + '-'
    full_config_name = full_config_name[:-1]
    full_config_name = str(abs(hash(full_config_name)) % (10 ** 8))
    result_path = directory_results + full_config_name
    metrics = test_attack_median_smoothing(model, id_test_loader, ood_test_loader, attack_name, attack_loss, punished_measure, start_data, magnitude, bounds, result_path)

    results = {
        'result_path': result_path,
        'fail_trace': seml.evaluation.get_results
    }

    return {**results, **metrics}
